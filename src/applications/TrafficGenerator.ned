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

package myprojectandtutorial.applications;
import inet.applications.udpapp.UdpBasicApp;

//
// Traffic Generator module.
//
// This class is responsible for generating all the network traffic used
// within the simulation. Each packet sent is a modelled as a "task" that
// is wrapped within a UDP packet. Each task contains a randomly generated number
// of CPU cycles required for process, which is bounded by a lower and upper value
// set within the omnetpp.ini file.
//
simple TrafficGenerator extends UdpBasicApp
{
    parameters:
        double minRequiredCPUCycles @units(Hz) = default (-1);
        double maxRequiredCPUCycles @units(Hz) = default (-1);
        @display("i=block/app");

    @class(TrafficGenerator);
}
