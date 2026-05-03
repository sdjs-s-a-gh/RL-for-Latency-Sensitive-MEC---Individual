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

Define_Module(ResourceAllocatorApp);

/**
 * Sets the fundamental settings for the simulation to proceed.
 *
 * This subroutines handles setting the MEC server's specifications and
 * ports to enable communication and resource allocation to occur. Logging
 * statistics also are initialised for the evaluation metrics to be tracked.
 * Lastly, the subroutine additionally handles creating the Python interpreter
 * should the user choose PPO as the allocation algorithm.
 *
 */
void ResourceAllocatorApp::initialize(int stage)
{
    ApplicationBase::initialize(stage);
    maxCPUCapacity = par("maxCPUCapacity").intValue();
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

            // Prior to importing the Python file, the system must be setup to be able to view the file.
            py::module_ sys = py::module_::import("sys");

            // Add the libraries used in the Python-side code so that PyTorch (for instance) may be accessed.
            sys.attr("path").attr("append")("/home/opp_env/.venv/lib/python3.12/site-packages");

            // Append the current directory (which to Pybind is "simulations") to be able to access the local script.
            sys.attr("path").attr("append")(".");

            // Finally, import the PyTorch resource allocator.
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

        // Setup connection to the cloud server.
        EV << "Dest Address:" << destAddressStr << endl;
        L3Address result;
        L3AddressResolver().tryResolve(destAddressStr.c_str(), result);

        EV << "Dest Address Again: " << result << endl;
        destAddress = result;

        EV << "My address: " << L3AddressResolver().resolve(getParentModule()->getFullPath().c_str()) << endl;

        // Without binding sockets, this app will only return a ICMP error that
        // indicates the incoming packet cannot reach the required socket - as the
        // application is yet to be on a socket.
        socket.setOutputGate(gate("socketOut"));
        socket.bind(localPort);
        socket.setCallback(this);
        EV << "EdgeServerResourceAllocatorApp has been successfully set up on Port: " << localPort << endl;
    }
}

/**
 * Receives and handles incoming messages to the MEC server.
 *
 * This subroutine routes an incoming message to another function for respective processing.
 * Once those functions have computed, the queue is then updated to determine whether the next
 * task can be processed.
 *
 * However, an erroneous message (caused by unknown means) will result in the simulation terminating
 * immediately.
 */
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
 * Allocates CPU Frequency to a particular task.
 *
 * This subroutine extracts a task's required CPU cycles for processing and routes
 * the value to the user's chosen allocation algorithm. The received CPU frequency allocation
 * is then assigned to the task along with its execution time.
 *
 * Should the user have chosen an invalid allocator algorithm, the simulation will terminate and
 * an error will be presented.
 */
void ResourceAllocatorApp::allocateResources(Task *task)
{
    // Extract CPU cycle from the task.
    int requiredCycles = task->requiredCPUCycles;

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
 * Returns the allocated CPU frequency of a given task based off a static ruleset.
 *
 * This function allocates an incoming task the equivalent frequency as the CPU Cycles required
 * for processing. For instance, a task that requires 100 Mcycles will be given 100MHz frequency.
 */
int ResourceAllocatorApp::staticAllocation(int requiredCycles)
{
    // The allocator just gives the incoming task the equivalent frequency as the cycles it requires.
    return requiredCycles;
}

/**
 * Returns a random CPU frequency for a given task completely irrespective of the CPU cycles required
 * to process the task.
 *
 * This function allocates an incoming task a random value between 1% and 100% of the CPU's max capacity
 * (to avoid some tasks taking forever).
 */
int ResourceAllocatorApp::randomAllocation()
{
    int lowerBound = maxCPUCapacity * 0.01;
    int randomCPUFrequency = intuniform(lowerBound, maxCPUCapacity);

    return randomCPUFrequency;
}

/**
 * Returns the allocated CPU frequency for a given task using the PPO Reinforcement Learning method.
 *
 * This function allocates an incoming task by communicating with a PPO algorithm in Python using Pybind.
 * The frequency allocation returned is based on the current observed state of the simulation at that time and
 * PPO's learned policy.
 *
 * If a Python or Pybind issue occurs, the simulation will terminate and a message will be shown to the user.
 */
int ResourceAllocatorApp::PPOAllocation(Task *task)
{
    // Get State and normalise to improve learning stability.
    double requiredCycles = task->requiredCPUCycles / (700.0); // 700 = Max CPU cycles set in ini.
    double communicationLatency = (task->communicationDelay.dbl() * 1000) / 50.0; // Convert to milliseconds
    double resourceUtilisation = getResourceUtilisation();
    double queueLength = (double) queue.getLength() / (double) maxQueueLength;
    double totalQueueCycles = (double) getTotalCyclesInQueue() / (700.0 * (double) maxQueueLength);

    // Combine the state into a vector, which Pybind will convert into a Python list.
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

        int allocatedCPUCycles = action * maxCPUCapacity;

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
 * Returns the amount of time (in seconds) executing a task of a certain size will take based on the capacity of
 * the MEC server.
 *
 * This function handles the execution model used in the simulation. The equation used to calculate the execution
 * time is given in section 3.1 of the dissertation.
 */
double ResourceAllocatorApp::getTimeToExecute(double cpuCyclesRequired, double allocatedCPUFrequency)
{
    return cpuCyclesRequired / allocatedCPUFrequency;
}

/**
 * Returns the current resource utilisation of the MEC server as a ratio.
 */
double ResourceAllocatorApp::getResourceUtilisation()
{
    return 1.0 - ((double) currentCapacity / (double) maxCPUCapacity);
}

/**
 * Returns the number of CPU cycles waiting to be processed in the current queue.
 *
 * This function iterates through each task in the queue and returns the sum of
 * all required CPU cycles waiting to be processed.
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
 * "Processes" a task in the simulation.
 *
 * As literal task processing does not occur in the simulation, this handles the appropriate
 * actions that are required to mimic task execution. Firstly, a self-message is scheduled to
 * a arrive back at the MEC when a task is set to be finished. Secondly, during this time,
 * the current capacity of the MEC server is decreased by the CPU cycles of a task to remove
 * available resources.
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

    // Increment the number of tasks currently being processed for debugging statistics.
    tasksProcessing++;
    EV << "Tasks Currently Processing: " << tasksProcessing << endl;
}

/**
 * A subroutine that finishes the execution of a task, freeing-up
 * CPU space.
 *
 * Finishes the execution of a task in the simulation.
 *
 * This subroutine is responsible for handling the self-messages used to indicate
 * the end of a task's execution. The subroutine includes calculating a task's energy consumption
 * before finalising all the evaluation metrics.
 *
 * If the PPO allocation algoriothm has been chosen by the user, the trajectory of a task (its lifecycle)
 * is sent to the agent to inform later training. The simulation will additionally terminate if the tasks
 * processed reaches the user-defined training episode lengths.
 *
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
 *
 * Handles processing the next task in the queue.
 *
 * This subroutine checks whether there is enough CPU frequency avaiable to process
 * the task at the head of the queue. If there is sufficient space, each task at the head
 * of the queue will be sent for processing. Otherwise, the simulation will continue on.
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

/*
 * Handles each packet that arrives on the MEC server.
 *
 * This subroutine is indirectly called by handleMessageWhenUp and is responsible
 * for the routing a task throughout the first part of its lifecyle in the
 * simulation. This journey firsly included offloading a tasks to the cloud server
 * should the queue be at capacity. Otherwise, they are allocated a CPU frequency
 * by whichever algorithm is selected by the user and subsequently added to the queue.
 * The handleMessageWhenUp() subroutine then checks the queue to determine whether
 * the task(s) at the head can be processed.
 */
void ResourceAllocatorApp::socketDataArrived(UdpSocket *socket, Packet *packet)
{
    packetsReceived++;

    EV << "========"<< endl << "socketDataArrived" << endl;
    EV << "Packet received: " << packet->getName() << ", size: " << packet->getByteLength() << " bytes" << endl;

    // Extract the CPU cycles from the packet.
    auto taskRequirements =  packet->peekAtFront<MyTaskChunk>();
    auto cpuCycles = taskRequirements->getRequiredCPUCycles();

    EV << "That was Packet " << packetsReceived << ". Required CPU Cycles: " << cpuCycles << endl;

    // If the queue is equal to its maximum length, delete/drop the incoming task.
    if (queue.getLength() >= maxQueueLength) {
        EV << "Queue is too big. Dropped Task." << endl;

        // Copy the packet as the original is destroyed at the end of the parent subroutine.
        Packet *copy = packet->dup();

        // Send the packet to the cloud server for external processing (which isn't done in this
        // simulation as discussed in section 4.1.).
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

/**
 * Handle the end of a simulation.
 *
 * This subroutine is triggered whenever a simulation ends. Namely, the method alters the inherited
 * behaviour by calling the PPO agent to update should the simulation end properly before subsequently
 * shutting down the interpreter agent. This last part is important since major errors occur as the
 * simulation cannot delete the interpreter properly otherwise. TODO: alter this last phrasing.
 */
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
