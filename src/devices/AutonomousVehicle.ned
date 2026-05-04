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

package myprojectandtutorial.devices;

import inet.node.inet.WirelessHost;

//
// Autonomous Vehicle device module.
//
// This module represents an autonomous vehicle used within the
// the simulation. There is no new code added or behaviour added
// here; instead, the only change is using a car icon to represent
// the device.
//
module AutonomousVehicle extends WirelessHost
{
    parameters:
        @display("i=misc/car3"); // Change the icon to a car.
}