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

#ifndef __MYPROJECTANDTUTORIAL_RESOURCEALLOCATORAPP_H_
#define __MYPROJECTANDTUTORIAL_RESOURCEALLOCATORAPP_H_

#include <omnetpp.h>
#include <inet/applications/base/ApplicationBase.h>
#include "inet/transportlayer/contract/udp/UdpSocket.h"
#include "MyTaskChunk_m.h"

using namespace omnetpp;
using namespace inet;

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
namespace py = pybind11;

/**
 * Better than a class since I need to access and modify everything.
 */
struct Task : public cObject
{
    int requiredCPUCycles; // The number of CPU cycles required to process/compute a task.
    int allocatedCPUFrequency; // The number of CPU cycles that will be processed in a single second - the speed of processing the task.
    double executionTime; // The time it will take for the task to be processed on the edge server. This variable is to be used for calculating when a self-message should be scheduled.


    // Variables Mainly for Statistics
    simtime_t creationTime; // The time at which the task was created and sent to the edge server from the mobile device.
    simtime_t completionTime; // The time at which the task finished executing on the edge server.

    simtime_t arrivalTime; // The time at which the task entered the edge server - just before it is allocated resources.
    simtime_t totalServiceTime; // In seconds, the total duration of the task's queueing + execution/processing time.
    simtime_t communicationDelay; // In seconds, the difference in time between the task being created and arriving at the edge server.

    simtime_t startQueueTime; // The time at which the task entered the queue.
    simtime_t endQueueTime; // The time at which the task left the queue.
    simtime_t totalQueueTime; // In seconds, the total duration of time the task was in the queue.

    simtime_t latency; // In seconds, the difference in time between the task being sent to the edge server and it finishing being processed.
    double energyConsumption; // The amount of energy (in an undefined unit) that was consumed to process the task.

    // Resource Allocation Specific
    std::vector<double> state; // The state space used to inform the CPU frequency that was to be allocated to the task.
    double logProbability; // The log probability given to the task for the action taken (CPU frequency allocated).
    double rawAction; // The raw action PPO produced prior to constraints.
};


/**
 * TODO - Generated class
 */
class ResourceAllocatorApp : public ApplicationBase, UdpSocket::ICallback
{
  protected:
    // Resource Allocation
    int64_t maxCPUCapacity = 10000; // in MHz currently
    int64_t currentCapacity = 0;
    cQueue queue;
    int resourceAllocatorAlgorithm = 0;

    int episodeLength; // The number of time-steps the episode will take to finish.

    // Python/Pybind-related
    bool pythonAllocatorStarted = false;
//    std::unique_ptr<py::scoped_interpreter> guard;
//    py::object agent;

    // Parameters
    int localPort = -1;
    L3Address destAddress;
    std::string destAddressStr;
    int destPort = -1;

    int maxQueueLength = 50;

    // Statistics
    int packetsReceived = 0;
    int tasksProcessed = 0;
    int tasksProcessing = 0;
    int cloudOffloadedTasks = 0;

    simsignal_t latencySignal;
    simsignal_t resourceUtilisationSignal;
    simsignal_t energyConsumptionSignal;
    simsignal_t tasksProcessedSignal;
    simsignal_t queueLengthSignal;
    simsignal_t parallelTasksSignal;
    simsignal_t cloudOffloadedTasksSignal;

    UdpSocket socket; // Requires a socket to bind the application to.

    void processTask(Task *task);
    double getTimeToExecute(double cpuCyclesRequired, double allocatedCPUCycles);
    void allocateResources(Task *task);
    void endTaskExecution(cMessage *msg);
    void updateQueue();
    double getResourceUtilisation();
    int getTotalCyclesInQueue();

    double calculateReward(double latency);

    void sendPacket(Packet * packet);

    // Allocation Algorithms
    int staticAllocation(int requiredCycles);
    int PPOAllocation(Task *task);
    int randomAllocation();

    virtual void finish() override;
    virtual void initialize(int stage) override;
    virtual void handleMessageWhenUp(cMessage *msg) override;

    virtual void refreshDisplay() const override;

    // Inheriting UdpSocket::ICallback requires these 3 methods.
    virtual void socketDataArrived(UdpSocket *socket, Packet *packet) override;
    virtual void socketErrorArrived(UdpSocket *socket, Indication *indication) override;
    virtual void socketClosed(UdpSocket *socket) override;

    // Inheriting ApplicationBase requires these 3 methods.
    virtual void handleStartOperation(LifecycleOperation *operation) override;
    virtual void handleStopOperation(LifecycleOperation *operation) override;
    virtual void handleCrashOperation(LifecycleOperation *operation) override;

  public:
    virtual int numInitStages() const override { return NUM_INIT_STAGES; }
};

#endif
