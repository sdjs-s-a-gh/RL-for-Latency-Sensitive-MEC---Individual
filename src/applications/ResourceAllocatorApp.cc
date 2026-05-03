//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
// 
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see http://www.gnu.org/licenses/.
// 

#include "ResourceAllocatorApp.h"
#include "MyTaskChunk_m.h"
#include "inet/networklayer/common/L3AddressResolver.h"
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
namespace py = pybind11;

static py::scoped_interpreter* guard = nullptr;
static py::object agent;

//#include <pybind11/embed.h>
// located at: home\opp_env\.venv\lib\python3.12 or /home/opp_env/.venv/lib/python3.12/site-packages
// stable: -I/usr/include/python3.12 -I/home/opp_env/.venv/lib/python3.12/site-packages/pybind11/include
// 3rd try: -L/nix/store/8w718rm43x7z73xhw9d6vh8s4snrq67h-python3-3.12.10/lib -lpython3.12 -ldl -L/nix/store/qizipyz9y17nr4w4gmxvwd3x4k0bp2rh-libxcrypt-4.4.38/lib -lm

Define_Module(ResourceAllocatorApp);

void ResourceAllocatorApp::initialize(int stage)
{
    ApplicationBase::initialize(stage);
    // Without binding sockets, this app will only return a ICMP error that
    // indicates the incoming packet cannot reach the required socket - as the
    // app is yet to be on a socket.

    maxCPUCapacity = par("maxCPUCapacity").intValue(); // Scale from MHz to normal Hz.
    currentCapacity = maxCPUCapacity;
    resourceAllocatorAlgorithm = par("resourceAllocatorAlgorithm");
    episodeLength = par("episodeLength");
    maxQueueLength = par("maxQueueLength");

    if (resourceAllocatorAlgorithm == 1 && pythonAllocatorStarted == false) {
        try {
            // Check to see if the interpreter is currently running, as it persists between simulation configurations.
            // Originally, this code did not need to exist as the interpreter persisted as a static object, but that caused
            // memory issues when increasing the number of mobile devices.
            if (!guard) {
                guard = new py::scoped_interpreter();  // Start the interpreter
            }

            // Import the Python File

            py::module_ sys = py::module_::import("sys");
            // TODO: Remove these comments : 1. Force the interpreter to look at your VENV's libraries
            sys.attr("path").attr("append")("/home/opp_env/.venv/lib/python3.12/site-packages");

            // 2. Also keep the current directory for your local script
            sys.attr("path").attr("append")(".");

            py::module_ rl_mod = py::module_::import("rl_resource_allocator");

            int stateSpaceDimensions = 5;
            int actionSpaceDimensions = 1;
            agent = rl_mod.attr("PPO")(stateSpaceDimensions, actionSpaceDimensions);

            py::print("Hello from Python inside OMNeT++!");
            pythonAllocatorStarted = true;

        } catch (py::error_already_set &e) {
            throw cRuntimeError("Python Error: %s", e.what());
        }
    }

    // Setup Statistics
    latencySignal = registerSignal("latency");
    resourceUtilisationSignal = registerSignal("resourceUtilisation");
    energyConsumptionSignal = registerSignal("energyConsumption");

    tasksProcessedSignal = registerSignal("tasksProcessed");
    queueLengthSignal = registerSignal("queueLength");
    parallelTasksSignal = registerSignal("parallelTasks");
    cloudOffloadedTasksSignal = registerSignal("cloudOffloadedTasks");

    if (stage == INITSTAGE_APPLICATION_LAYER) {
        localPort = par("localPort");
        destPort = par("destPort");
        destAddressStr = par("destAddress").stdstringValue();

        EV << "Dest Address:" << destAddressStr << endl;
        L3Address result;
        L3AddressResolver().tryResolve(destAddressStr.c_str(), result);

        EV << "Dest Address Again: " << result << endl;
        destAddress = result;

        EV << "My address: " << L3AddressResolver().resolve(getParentModule()->getFullPath().c_str()) << endl;

        socket.setOutputGate(gate("socketOut"));
        socket.bind(localPort);
        socket.setCallback(this);
        EV << "EdgeServerResourceAllocatorApp has been successfully set up on Port: " << localPort << endl;
    }
}

void ResourceAllocatorApp::handleMessageWhenUp(cMessage *msg)
{
    // If a task has completed.
    if (msg->isSelfMessage() && strcmp(msg->getName(), "endExecutionSelfMessage") == 0) {
        endTaskExecution(msg);
    // Incoming Task
    } else if (msg->arrivedOn("socketIn")) {
        // Process a Packet
        socket.processMessage(msg);
    } else {
        throw cRuntimeError("Unknown message received on gate: %s", msg->getArrivalGate()->getFullName());
    }

    // Regardless of the message type (not providing an erroneous type), update the queue
    // to check whether the next task can be processed.
    updateQueue();
    delete msg;
}

/**
 * Allocates CPU cycles to the task passed as an argument.
 * The subroutine works by first routing the chosen allocation method to
 * its corresponding function.
 */
void ResourceAllocatorApp::allocateResources(Task *task)
{
    int requiredCycles = task->requiredCPUCycles;
    double communicationLatency = (task->communicationDelay.dbl() * 1000); // Convert to milliseconds

    int cpuCyclesToAllocate = 0;
    if (resourceAllocatorAlgorithm == 0) {
        cpuCyclesToAllocate = staticAllocation(requiredCycles);
    } else if (resourceAllocatorAlgorithm == 1) {
        cpuCyclesToAllocate = PPOAllocation(task);
    } else if (resourceAllocatorAlgorithm == 2){
        cpuCyclesToAllocate = randomAllocation();
    } else {
        throw cRuntimeError("You've given an invalid resource allocator algorithm: %d", resourceAllocatorAlgorithm);
    }

    task->allocatedCPUFrequency = cpuCyclesToAllocate;
    task->executionTime = getTimeToExecute(task->requiredCPUCycles, cpuCyclesToAllocate);

}

/**
 * Returns the allocated CPU frequency of a given task based off a static ruleset. The algorithm achieves this by
 * allocating the incoming task whatever cycles it needs.
 */
int ResourceAllocatorApp::staticAllocation(int requiredCycles)
{
    // TODO: At the minute, the allocator just gives the incoming task whatever cycles it needs.
    return requiredCycles;
}

/**
 * Returns a random CPU frequency for a given task completely irrespective of the size (CPU cycles are needed)
 * to process the task. The number returned will be between 1% and 100% of the CPU's max capacity to avoid
 * some tasks taking forever.
 */
int ResourceAllocatorApp::randomAllocation()
{
    int lowerBound = maxCPUCapacity * 0.01;
    int randomCPUFrequency = intuniform(lowerBound, maxCPUCapacity);

    return randomCPUFrequency;
}

/**
 * Returns the allocated CPU frequency for a given task using the Reinforcement Learning method
 * of PPO.
 */
int ResourceAllocatorApp::PPOAllocation(Task *task)
{
    // Get State and normalise to improve learning stability. // TODO
    double requiredCycles = task->requiredCPUCycles / (700.0); // 700 = Max CPU cycles set in ini.
    double communicationLatency = (task->communicationDelay.dbl() * 1000) / 50.0; // Convert to milliseconds
    double resourceUtilisation = getResourceUtilisation();
    double queueLength = (double) queue.getLength() / (double) maxQueueLength;
    double totalQueueCycles = (double) getTotalCyclesInQueue() / (700.0 * (double) maxQueueLength);

    std::vector<double>state = {
            requiredCycles,
            communicationLatency,
            resourceUtilisation,
            queueLength,
            totalQueueCycles
    };

    EV << "State: " << requiredCycles << ", " << communicationLatency << ", "
       << resourceUtilisation << ", " << queueLength << ", " << totalQueueCycles << endl;

    try {
        py::tuple result = agent.attr("get_action")(state);
        double rawAction = result[0].cast<double>();
        double logProbability = result[1].cast<double>();

        // Clip/Constrain the action to be between [0.01, 1.0].
        double action = std::clamp(abs(rawAction), 0.01, 1.0);

        EV << "Action: " << action << "; Log Prob: " << logProbability << "; Raw Action: " << rawAction << endl;

        int allocatedCPUCycles = action * maxCPUCapacity; // TODO: Will need to round to the nearest int.

        EV << "Max CPU Capacity: " << maxCPUCapacity << endl;

        EV << "The RL Agent has decided to allocate " << action << " cycles as a ratio or "<< allocatedCPUCycles << " cycles." << endl;

        // Save most of the trajectory in the task for future reference for when the task has finished executing (the time-step has finished).
        task->state = state;
        task->logProbability = logProbability;
        task->rawAction = rawAction;

        return allocatedCPUCycles;

    } catch (py::error_already_set &e){
        throw cRuntimeError("Python Error: %s", e.what());
    };
}

/**
 * Returns the amount of time in seconds it will take to
 * execute a task of a certain size based on the capacity of
 * the edge server.
 */
double ResourceAllocatorApp::getTimeToExecute(double cpuCyclesRequired, double allocatedCPUFrequency)
{
    return cpuCyclesRequired / allocatedCPUFrequency; //maxCPUCapacity, allocatedCPUFrequency
}

/**
 * Returns the current resource utilisation of the edge server - between 0 and 1.
 */
double ResourceAllocatorApp::getResourceUtilisation()
{
    return 1.0 - ((double) currentCapacity / (double) maxCPUCapacity);
}

/**
 * Returns the number of CPU cycles needing to be processed in the current queue.
 */
int ResourceAllocatorApp::getTotalCyclesInQueue()
{
    int totalCycles = 0;

    // Iterate through each task in the queue
    for (cQueue::Iterator iter(queue); !iter.end(); iter++) {
        auto *task = check_and_cast<Task *>(*iter);

        totalCycles += task->requiredCPUCycles;
    }

    return totalCycles;
}

/**
 * The subroutine that processes the input task.
 *
 * The subroutine takes the task to be executed on the CPU
 * and subsequently deallocates the CPU cycles required from
 * the current capacity of the edge server.
 */
void ResourceAllocatorApp::processTask(Task *task)
{
    EV << "========" << endl << "processTask" << endl;
    auto *endExecutionSelfMessage = new cMessage("endExecutionSelfMessage");
    endExecutionSelfMessage->setContextPointer(task);

    scheduleAt(simTime() + task->executionTime, endExecutionSelfMessage);
    currentCapacity -= task->allocatedCPUFrequency;

    simtime_t scheduleTimeTotal = simTime() + task->executionTime;
    simtime_t scheduleTimeInMS = scheduleTimeTotal * 1000;

    EV << "Processing Task. Current Capacity: " << currentCapacity << endl;
    EV << "That will take " << task->executionTime << " seconds. According to the simulator, that will be at "
            << SIMTIME_STR(scheduleTimeTotal) << ", which is at " << scheduleTimeInMS << "ms." << endl;
    tasksProcessing++;
    EV << "Tasks Currently Processing: " << tasksProcessing << endl;
}

/**
 * A subroutine that finishes the execution of a task, freeing-up
 * CPU space.
 */
void ResourceAllocatorApp::endTaskExecution(cMessage *msg)
{
    EV << "========" << endl << "endTaskExecution" << endl;
    EV << "Received Self-Message. Finishing task. ";

    auto *completedTask = static_cast<Task *>(msg->getContextPointer());

    completedTask->completionTime = simTime();
    completedTask->totalServiceTime = completedTask->completionTime - completedTask->arrivalTime;
    completedTask->latency = completedTask->completionTime - completedTask->creationTime;

    double k_Scaled = 1e-27 * 1e+18;
    double energyConsumption = k_Scaled * completedTask->requiredCPUCycles * (completedTask->allocatedCPUFrequency * completedTask->allocatedCPUFrequency);
    completedTask->energyConsumption = energyConsumption;

    if (resourceAllocatorAlgorithm == 1) {
        std::vector state = completedTask->state;
        double action = completedTask->rawAction;
        double logProbability = completedTask->logProbability;
        double latency = completedTask->latency.dbl() * 1000;
        double resourceUtilisation = getResourceUtilisation();

        // Get the current queue utilisation to punish for being too big and offloading tasks.
        // TODO: Create a function to return the queue utilisation considering I already use this when
        // getting an action from the PPO agent.
        double queueUtilisation = (double) queue.getLength() / (double) maxQueueLength;

        EV << "Latency: " << latency << "; State: "  << "; Action: " << action << "; Log Prob: " << logProbability << endl;
        EV << "Energy Consumption: " << energyConsumption << endl;

        // Send the details of the trajectory to the PPO agent.
        try {
            agent.attr("add_trajectory")(state, action, logProbability, latency, energyConsumption, queueUtilisation);

        } catch (py::error_already_set &e){
            throw cRuntimeError("Python Error: %s", e.what());
        }
    }

    // Free up the CPU of the edge server by giving back the allocated CPU cycles.
    currentCapacity += completedTask->allocatedCPUFrequency;
    tasksProcessing--;
    tasksProcessed++;

    EV << "Capacity is now: " << currentCapacity << endl;
    // These print statements could be inside a function of Task instead.
    EV << "Task was Created: " << SIMTIME_STR(completedTask->creationTime) << "; Completion Time: " << SIMTIME_STR(completedTask->completionTime) << endl;
    EV << "Began Servicing: " << SIMTIME_STR(completedTask->arrivalTime) << "; Finished Servicing: " << SIMTIME_STR(completedTask->completionTime) << endl;
    EV << "The total service time for that task was: " << completedTask->totalServiceTime << " seconds." << endl;
    EV << "Tasks Processed: " << tasksProcessed << endl;
    EV << "Task Latency: " << completedTask->latency << " seconds, or " << completedTask->latency * 1000 << " ms." << endl;
    EV << "Task Energy Consumption: " << completedTask->energyConsumption << endl;
    EV << "Task CPU Cycles Required: " << completedTask->requiredCPUCycles << "; Expected Execution Time: " << getTimeToExecute(completedTask->requiredCPUCycles, completedTask->requiredCPUCycles)
            << "; Actual Cycles Given: " << completedTask->allocatedCPUFrequency << "; Actual Execution Time: " << getTimeToExecute(completedTask->requiredCPUCycles,completedTask->allocatedCPUFrequency) << endl;

    emit(latencySignal,completedTask->latency * 1000);
    emit(energyConsumptionSignal, energyConsumption);
    emit(tasksProcessedSignal, tasksProcessed);
    emit(parallelTasksSignal, tasksProcessing);

    if (tasksProcessed >= episodeLength) {
        // End the Simulation
        endSimulation();
    }
}

/**
 * A subroutine that checks whether the next task in the queue can be processed.
 */
void ResourceAllocatorApp::updateQueue()
{
    EV << "========" << endl << "updateQueue" << endl;

    if (!queue.isEmpty()) {
        auto *queueHead = check_and_cast<Task *>(queue.front());

        EV << "Is the Queue Empty? " << queue.isEmpty() << endl;
        EV << "Current Capacity: " << currentCapacity << endl;

        // TODO: Need to fix the queue as it will eventually break if there is nothing in it.
        // Keep processing tasks while there is enough resources remaining.
        while (!queue.isEmpty() && queueHead->allocatedCPUFrequency <= currentCapacity) {
            processTask(queueHead);

            // Only now remove (dequeue) the task from the queue once there are enough resources.
            queue.pop();
            }
        EV << "The length of the queue is: " << queue.getLength() << endl;
        emit(queueLengthSignal, queue.getLength());
        EV << "Resource Utilisation: " << getResourceUtilisation() << endl;
        emit(resourceUtilisationSignal, getResourceUtilisation());
    }

}


/**
 * =========================================================================
 * From UdpSocket::ICallback
 */
void ResourceAllocatorApp::socketDataArrived(UdpSocket *socket, Packet *packet)
{
    packetsReceived++;

    EV << "========"<< endl << "socketDataArrived" << endl;
    EV << "Packet received: " << packet->getName() << ", size: " << packet->getByteLength() << " bytes" << endl;

    auto taskRequirements =  packet->peekAtFront<MyTaskChunk>();
    auto cpuCycles = taskRequirements->getRequiredCPUCycles();

    EV << "That was Packet " << packetsReceived << ". Required CPU Cycles: " << cpuCycles << endl;

    // TODO:
    // If the queue is over 50 tasks long, delete/drop the incoming task.
    if (queue.getLength() >= 50) {
        EV << "Queue is too big. Dropped Task." << endl;

        // Copy the packet as the original is destroyed at the end of the parent subroutine.
        Packet *copy = packet->dup();
        sendPacket(copy);
        return;
    }

    // Retrieve the task requirements from the Packet's payload.
    auto taskChunk = packet->peekAtFront<MyTaskChunk>();

    // As Chunks are immutable, copy the data over from them into a new variable.
    auto *task = new Task();
    task->requiredCPUCycles = taskChunk->getRequiredCPUCycles();

    task->arrivalTime = simTime();
    task->communicationDelay = task->arrivalTime - taskChunk->getCreationTime();
    task->creationTime = taskChunk->getCreationTime();

    EV << "Communication Delay: " << task->communicationDelay << endl;

    allocateResources(task);
    EV << "CPU Cycles Allocated: " << task->allocatedCPUFrequency << endl;
    queue.insert(task); // enqueue the task
}

/**
 * Forwards a packet to the cloud server - or whichever destination address has been set
 * in the .ini file.
 */
void ResourceAllocatorApp::sendPacket(Packet *packet)
{
    EV << "Dest address to send to: " << destAddress << "; Port: " << destPort << endl;

    packet->clearTags();
    socket.sendTo(packet, destAddress, destPort);

    cloudOffloadedTasks++;
    emit(cloudOffloadedTasksSignal,cloudOffloadedTasks);
}

void ResourceAllocatorApp::refreshDisplay() const
{
    ApplicationBase::refreshDisplay();

    char buf[50];
    sprintf(buf, "Packets Received: %d", packetsReceived);
    getDisplayString().setTagArg("t", 0, buf);
}

void ResourceAllocatorApp::socketErrorArrived(UdpSocket *socket, Indication *indication)
{
    EV_WARN << "Ignoring UDP error report " << indication->getName() << endl;
    delete indication;
}

void ResourceAllocatorApp::socketClosed(UdpSocket *socket)
{

}

/**
 * ==========================================================================
 * From Application Base
 */
void ResourceAllocatorApp::handleStartOperation(LifecycleOperation *operation)
{

}

void ResourceAllocatorApp::handleStopOperation(LifecycleOperation *operation)
{

}

void ResourceAllocatorApp::handleCrashOperation(LifecycleOperation *operation)
{

}

void ResourceAllocatorApp::finish()
{
    // TODO: Tidy this subroutine up.
    EV << "Tasks Processed: "<< tasksProcessed << endl;

    if (tasksProcessed == episodeLength) {
        if (resourceAllocatorAlgorithm == 1) {
            try {
                // Flush the print statements out since they don't carry over here automatically anymore.
                py::exec("import sys; sys.stdout.flush(); sys.stderr.flush()");

                 // Tell the Resource Allocator to update as the episode has ended.
                //agent.attr("update_and_save")();

                EV << "The agent should have saved by now." << endl;

             } catch (py::error_already_set &e){
                 throw cRuntimeError("Python Error: %s", e.what());
             }
             EV << "The Simulation has finished gracefully." << endl;
        }
        else {
            EV << "The Simulation has finished gracefully." << endl;
        }
    } else {
        EV << "The Simulation has finished prematurely." << endl;
    }

    agent = py::none();
    cSimpleModule::finish();
}
