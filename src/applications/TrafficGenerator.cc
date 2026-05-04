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

#include "TrafficGenerator.h"
#include "MyTaskChunk_m.h"
#include "inet/applications/base/ApplicationPacket_m.h"
#include "inet/networklayer/common/FragmentationTag_m.h"
#include "inet/common/TimeTag_m.h"

Define_Module(TrafficGenerator);

/**
 * Creates and sends a packet towards a specified device.
 *
 * This subroutine creates a UDP data packet to be sent towards
 * the MEC server for processing. The code first calculates the
 * task size (CPU cycles required) by sampling a random value
 * from a uniform distribution, which is bounded by an upper and
 * lower boundary set in the omnetpp.ini file. Additionally, the
 * creation time of the packet is additionally recorded to later
 * compute the task completion latency.
 */
void TrafficGenerator::sendPacket()
{
    // Create the Container to send the data (packet).
    std::ostringstream str;
    str << packetName << "-" << numSent;
    Packet *packet = new Packet(str.str().c_str());

    if (dontFragment)
        packet->addTag<FragmentationReq>()->setDontFragment(true);

    // Create the data (Payload).
    const auto& payload = makeShared<MyTaskChunk>();

    // Sample a random value from a Uniform distribution.
    double cpuCycles = uniform(
            par("minRequiredCPUCycles").doubleValue(),
            par("maxRequiredCPUCycles").doubleValue()
            );

    payload->setRequiredCPUCycles(cpuCycles);
    payload->setCreationTime(simTime());

    EV << "The sim time is: " << simTime() << endl;
    EV << "The sim time for the payload is: " << payload->getCreationTime() << endl;

    payload->setChunkLength(B(par("messageLength")));
    packet->insertAtBack(payload);

    EV_INFO << "Sending chunk type: " << payload->getChunkType() << endl;

    L3Address destAddr = chooseDestAddr();

    socket.sendTo(packet, destAddr, destPort);
    numSent++;
}

/**
 * Calls the sendPacket() function.
 *
 * This subroutine is required by the parent class (UdpBasicApp)
 * to call the sendPacket() function specified above.
 */
void TrafficGenerator::processStart() {
    EV_INFO << "TrafficGenerator processStart called" << endl;
    UdpBasicApp::processStart(); // call base to schedule sending
    EV_INFO << "From: " << getFullPath() << endl;
}
