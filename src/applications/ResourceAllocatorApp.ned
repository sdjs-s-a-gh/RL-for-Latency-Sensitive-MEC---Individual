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
import inet.applications.contract.IApp;

//
// Resource Allocator application.
//
// This module acts as an application and is responsible for handling
// the allocation of MEC resources (CPU frequency) within the simulation.
//
// The module requires a total of 7 parameters in order to function://
// * 3 port-realted - which are essential for packets to reach the MEC server
// and the cloud server. <br>
// * 2 MEC-specific - which allow different MEC configurations to be tested; <br>
// * resourceAllocatorAlgorithm (int): The resource allocation algorithm to be used for
// the experiment. <br>
// * episodeLength (int): The episode length used for training the PPO algorithm - should the experiment
// use it.
//
simple ResourceAllocatorApp like IApp
{
	// All applications require IApp as an interface,
	// which requires an input and output set of sockets.
	parameters:
	    @display("i=block/app");	
	    int localPort;
	    string destAddress;
	    int destPort;
	
	    int maxCPUCapacity = default(10000);
	    int maxQueueLength = default(50);
	
	    int resourceAllocatorAlgorithm;
	
	    int episodeLength;

        @signal[latency](type=double);
        @signal[resourceUtilisation](type=double);
        @signal[energyConsumption](type=double);
        @signal[tasksProcessed](type=int);
        @signal[maxQueueLength](type=int);
        @signal[parallelTasks](type=int);
        @signal[cloudOffloadedTasks](type=int);
	
	gates:
	    input socketIn;
	    output socketOut;
}
