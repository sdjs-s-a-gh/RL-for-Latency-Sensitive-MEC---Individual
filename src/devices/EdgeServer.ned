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

import inet.node.inet.StandardHost;

//
// MEC Server device module.
//
// This module represents the MEC server used to process user tasks
// within the simulation. There is no new code added or behaviour added
// here; instead, the only change is using a server icon to represent
// the device.
//
module EdgeServer extends StandardHost
{
    parameters:
        @display("i=device/server"); // Change the icon to a server.
}
