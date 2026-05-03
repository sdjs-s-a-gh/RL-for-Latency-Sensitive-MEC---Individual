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

package myprojectandtutorial.simulations;

// Local Imports
import myprojectandtutorial.devices.EdgeServer;
import myprojectandtutorial.devices.MobilePhone;
import myprojectandtutorial.devices.AutonomousVehicle;
import myprojectandtutorial.devices.ARVRUser;
import myprojectandtutorial.devices.IoTDevice;
import myprojectandtutorial.devices.CloudServer;

// Wireless Connections
import inet.node.inet.WirelessHost;
import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.physicallayer.wireless.common.contract.packetlevel.IRadioMedium;
// Visualing the Traffic Propagate throughout the medium
import inet.visualizer.canvas.integrated.IntegratedCanvasVisualizer;

// Physical Connections
import inet.node.inet.StandardHost;
import inet.node.inet.Router;
import inet.node.ethernet.Eth10M;

//import inet.physicallayer.wireless.common.medium.RadioMedium; // Required for wireless

network ResourceNetwork
{
    parameters:
        @display("bgb=1000,1000");
        int numberOfMobilePhones = default(10);
        int numberOfAutonomousVehicles = default(7);
        int numberOfIoTDevices = default(5);
        int numberOfARVRUsers = default(3);

        // Statistics to be gathered during the simulation.
        // Main
        @statistic[latencyOverTime](source=latency; record=vector);
        @statistic[systemResourceUtilisationOverTime](source=resourceUtilisation; record=vector);
        @statistic[energyConsumptionOverTime](source=energyConsumption; record=vector);

        // Complementary
        @statistic[tasksProcessedOverTime](source=tasksProcessed; record=vector);
        @statistic[queueLengthOverTime](source=queueLength; record=vector);
        @statistic[parallelTasksOverTime](source=parallelTasks; record=vector);
        @statistic[cloudOffloadedTasksOverTime](source=cloudOffloadedTasks; record=vector);

    submodules:
        visualizer: IntegratedCanvasVisualizer {
            @display("p=62,870");
        }
        configurator: Ipv4NetworkConfigurator {
            @display("p=62,943");
        }
        radioMedium: <default("ApskScalarRadioMedium")> like IRadioMedium {
            @display("p=62,795");
        } // Will need to change to RadioMedium to configure bandwidth

        // User Layer
        mdMobilePhone[numberOfMobilePhones]: MobilePhone {
            @display("p=250,250");	// The initial position of the node will be overridden in the .ini file.
        }

        mdAutonomousVehicle[numberOfAutonomousVehicles]: AutonomousVehicle {
            @display("p=428,289");
        }

        mdARVRUser[numberOfARVRUsers]: ARVRUser {
            @display("p=110,129");
        }

        mdIoTDevice[numberOfIoTDevices]: IoTDevice {
            @display("p=598,195");
        }


        // Edge Layer
        baseStation: WirelessHost {
            @display("p=428,546");
            @display("i=device/antennatower");
        }
        coreRouter: Router {
            @display("p=428,475");
        }
        edgeServer: EdgeServer {
            @display("p=532,475");
        } // Was StandardHost

        // Cloud Layer
        cloudServer: CloudServer {
            @display("p=920,44");
        }


    // Physical Connections
    connections:
        baseStation.ethg++ <--> Eth10M <--> coreRouter.ethg++;
        coreRouter.ethg++ <--> Eth10M <--> edgeServer.ethg++;
        coreRouter.ethg++ <--> Eth10M <--> cloudServer.ethg++;	// Backhaul connection to a data centre (which isn't realisistic as it would go through many networks first).

}
