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
// Traffic Generator module application.
//
// This module acts as an application and is attached to the class responsible
// for generating all the network traffic used within the simulation. This application
// is to be used by the user devices to generate tasks that will be processed by the MEC
// server.
//
// Each task contains a randomly generated number of CPU cycles required
// for processing, which is bounded by a required lower and upper value
// that represents a CPU cycle given in MHz rather than Hz.
//
simple TrafficGenerator extends UdpBasicApp
{
    parameters:
        double minRequiredCPUCycles @units(Hz) = default (-1);
        double maxRequiredCPUCycles @units(Hz) = default (-1);
        @display("i=block/app");

    @class(TrafficGenerator);
}
