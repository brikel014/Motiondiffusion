


import carla
import time
import sys
import json
import random
import math
import xml.etree.ElementTree as ET
from xml.dom import minidom # For pretty printing XML
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Helper function to calculate boundary points ---
def calculate_boundary_points(waypoint):
    """Calculates the left and right boundary points for a given waypoint."""
    transform = waypoint.transform
    location = transform.location
    right_vector = transform.get_right_vector()
    half_width = waypoint.lane_width / 2.0
    left_boundary_loc = location - carla.Location(x=right_vector.x * half_width,
                                                  y=right_vector.y * half_width)
    right_boundary_loc = location + carla.Location(x=right_vector.x * half_width,
                                                   y=right_vector.y * half_width)
    return (left_boundary_loc.x, left_boundary_loc.y), \
           (right_boundary_loc.x, right_boundary_loc.y)

# --- Function to extract map boundary data ---
def extract_map_data(carla_map, segment_resolution=2.0):
    """
    Extracts the CARLA map topology boundary points.
    :param carla_map: The carla.Map object.
    :param segment_resolution: Distance between points along the lanelet boundaries (meters).
    :return: Tuple: (all_left_boundaries, all_right_boundaries, processed_segments_keys)
    """
    print(f"Extracting map topology points with resolution {segment_resolution}m...")
    topology = carla_map.get_topology()
    processed_segments = set()
    all_left_boundaries = []
    all_right_boundaries = []
    lanelet_counter = 0

    for start_wp, end_wp in topology:
        segment_key = (start_wp.road_id, start_wp.lane_id, start_wp.id)
        if segment_key in processed_segments:
            continue

        lanelet_counter += 1
        left_points_data = []
        right_points_data = []
        current_wp = start_wp
        try:
            left_pt, right_pt = calculate_boundary_points(current_wp)
            left_points_data.append(left_pt)
            right_points_data.append(right_pt)
            visited_wp_ids = {current_wp.id}
        except Exception as e:
             print(f"Warning: Error processing start waypoint {current_wp.id} for segment {segment_key}: {e}")
             continue


        while True:
            try:
                dist_to_end = current_wp.transform.location.distance(end_wp.transform.location)
                if dist_to_end < segment_resolution * 0.5:
                    break
                next_waypoints = current_wp.next(segment_resolution)
                if not next_waypoints:
                    break
                next_wp = next_waypoints[0]

                if next_wp.id in visited_wp_ids:
                    break

                dist_next_to_end = next_wp.transform.location.distance(end_wp.transform.location)
                if dist_next_to_end > dist_to_end and dist_to_end < segment_resolution :
                     break


                left_pt, right_pt = calculate_boundary_points(next_wp)
                left_points_data.append(left_pt)
                right_points_data.append(right_pt)
                visited_wp_ids.add(next_wp.id)
                current_wp = next_wp

                if len(visited_wp_ids) > 750:
                    print(f"Warning: Exceeded step limit (750) for segment trace {lanelet_counter}. Stopping trace.")
                    break
            except Exception as e:
                 print(f"Warning: Error during segment trace for key {segment_key} near waypoint {current_wp.id}: {e}")
                 break


        try:
            last_wp_loc = current_wp.transform.location
            if last_wp_loc.distance(end_wp.transform.location) > 1e-2 :
                if end_wp and end_wp.transform:
                    left_pt, right_pt = calculate_boundary_points(end_wp)
                    if not left_points_data or \
                    (abs(left_points_data[-1][0] - left_pt[0]) > 1e-3 or abs(left_points_data[-1][1] - left_pt[1]) > 1e-3):
                        left_points_data.append(left_pt)
                        right_points_data.append(right_pt)
                else:
                     print(f"Warning: End waypoint {end_wp.id} seems invalid for segment {segment_key}")
        except Exception as e:
             print(f"Warning: Error adding end waypoint {end_wp.id} for segment {segment_key}: {e}")


        if len(left_points_data) > 1 and len(right_points_data) > 1:
            all_left_boundaries.append(left_points_data)
            all_right_boundaries.append(right_points_data)
            processed_segments.add(segment_key)

    print(f"Extracted data for {len(all_left_boundaries)} valid lane segments.")
    return all_left_boundaries, all_right_boundaries, processed_segments


# --- Function to Write Combined XML (Map + Dynamic Obstacles) ---
def write_combined_xml(carla_map, all_left_boundaries, all_right_boundaries, processed_segments_keys,
                       initial_vehicle_actors_list, trajectory_data, output_filename="simulation_output.xml", time_step_output=0.2): # MODIFIED: initial_vehicle_actors_list
    """
    Writes map boundaries and vehicle trajectories (as dynamicObstacles) to XML.
    Outputs trajectory states with integer time steps, where each integer step
    represents 'time_step_output' seconds of real simulation time.

    :param carla_map: The carla.Map object.
    :param all_left_boundaries: List of lists of (x, y) left points.
    :param all_right_boundaries: List of lists of (x, y) right points.
    :param processed_segments_keys: Set of (road_id, lane_id, start_wp.id) keys.
    :param initial_vehicle_actors_list: List of ALL carla.Actor objects for vehicles SPAWNED in sim (used for bbox). # MODIFIED
    :param trajectory_data: Dictionary mapping vehicle IDs ("vehicle_XXX") to lists of state dicts FOR RECORDED VEHICLES.
                            State dicts must include 'timestamp', 'position', 'orientation_deg',
                            'velocity_mps' (tuple), and 'acceleration_mps2' (scalar).
    :param output_filename: Name of the output XML file.
    :param time_step_output: The desired REAL time increment between states (e.g., 0.2 seconds).
    """
    print(f"Writing combined map and dynamic obstacle data to {output_filename}...")
    print(f"Trajectory states will be output with integer time steps, each representing {time_step_output} seconds.")
    root = ET.Element("commonRoad")
    lanelet_id_counter = 0

    # --- 1. Write Lanelet Data ---
    segment_data_map = {}
    topology = carla_map.get_topology()
    for start_wp, _ in topology:
        segment_key = (start_wp.road_id, start_wp.lane_id, start_wp.id)
        if segment_key in processed_segments_keys:
             segment_data_map[segment_key] = start_wp

    processed_keys_list = sorted(list(processed_segments_keys))

    if len(processed_keys_list) != len(all_left_boundaries):
         print(f"Warning: Mismatch between processed keys ({len(processed_keys_list)}) and extracted boundaries ({len(all_left_boundaries)}). XML map data might be incomplete or misaligned.")

    num_boundaries = min(len(processed_keys_list), len(all_left_boundaries), len(all_right_boundaries))

    for i in range(num_boundaries):
        lanelet_id_counter += 1
        lanelet_elem = ET.SubElement(root, "lanelet", id=str(lanelet_id_counter))
        left_bound_elem = ET.SubElement(lanelet_elem, "leftBound")
        right_bound_elem = ET.SubElement(lanelet_elem, "rightBound")

        left_points_data = all_left_boundaries[i]
        right_points_data = all_right_boundaries[i]
        current_segment_key = processed_keys_list[i]

        for lx, ly in left_points_data:
            point_elem = ET.SubElement(left_bound_elem, "point")
            ET.SubElement(point_elem, "x").text = f"{lx:.4f}"
            ET.SubElement(point_elem, "y").text = f"{ly:.4f}"

        for rx, ry in right_points_data:
            point_elem = ET.SubElement(right_bound_elem, "point")
            ET.SubElement(point_elem, "x").text = f"{rx:.4f}"
            ET.SubElement(point_elem, "y").text = f"{ry:.4f}"

        start_wp = segment_data_map.get(current_segment_key)
        if start_wp:
            try:
                left_marking_type = start_wp.left_lane_marking.type
                right_marking_type = start_wp.right_lane_marking.type
                def map_marking(marking_type):
                     if marking_type == carla.LaneMarkingType.NONE: return "no_marking"
                     if marking_type == carla.LaneMarkingType.Other: return "other"
                     if marking_type == carla.LaneMarkingType.Broken: return "broken"
                     if marking_type == carla.LaneMarkingType.Solid: return "solid"
                     if marking_type == carla.LaneMarkingType.SolidSolid: return "solid_solid"
                     if marking_type == carla.LaneMarkingType.SolidBroken: return "solid_broken"
                     if marking_type == carla.LaneMarkingType.BrokenSolid: return "broken_solid"
                     if marking_type == carla.LaneMarkingType.BrokenBroken: return "broken_broken"
                     if marking_type == carla.LaneMarkingType.BottsDots: return "botts_dots"
                     if marking_type == carla.LaneMarkingType.Grass: return "grass"
                     if marking_type == carla.LaneMarkingType.Curb: return "curb"
                     return str(marking_type).lower()
                ET.SubElement(left_bound_elem, "lineMarking").text = map_marking(left_marking_type)
                ET.SubElement(right_bound_elem, "lineMarking").text = map_marking(right_marking_type)
            except Exception as e:
                 print(f"Warning: Error getting markings for segment {current_segment_key}: {e}")
                 ET.SubElement(left_bound_elem, "lineMarking").text = "error"
                 ET.SubElement(right_bound_elem, "lineMarking").text = "error"
        else:
             ET.SubElement(left_bound_elem, "lineMarking").text = "unknown"
             ET.SubElement(right_bound_elem, "lineMarking").text = "unknown"

    print(f"Added {lanelet_id_counter} lanelet elements to XML.")

    # --- 2. Write Dynamic Obstacle Data ---
    vehicle_map = {v.id: v for v in initial_vehicle_actors_list} # MODIFIED: Use initial list for bbox lookup

    if not trajectory_data:
        print("No trajectory data to add to XML.")
    else:
        print(f"Adding trajectory data for {len(trajectory_data)} dynamic obstacles to XML...")

    obstacle_count = 0
    for vehicle_id_str, states in trajectory_data.items():
        if not states or len(states) < 2: # Need at least initial state and one trajectory point for CR
             print(f"Skipping {vehicle_id_str} due to insufficient states ({len(states)}). Needs at least 2 (initial + 1).")
             continue

        try:
            numeric_id = int(vehicle_id_str.split('_')[-1])
            vehicle_actor = vehicle_map.get(numeric_id) # Look up in map of ALL spawned vehicles

            if not vehicle_actor: # Should not happen if trajectory_data keys are from valid actors
                 print(f"Warning: Could not find vehicle actor object for ID {numeric_id} in initial list. Skipping.")
                 continue

            obstacle_count += 1
            obs_elem = ET.SubElement(root, "dynamicObstacle", id=str(numeric_id)) # Use original CARLA ID

            ET.SubElement(obs_elem, "type").text = "car"
            try:
                 extent = vehicle_actor.bounding_box.extent
                 shape_elem = ET.SubElement(obs_elem, "shape")
                 rect_elem = ET.SubElement(shape_elem, "rectangle")
                 ET.SubElement(rect_elem, "length").text = f"{2.0 * extent.x:.4f}"
                 ET.SubElement(rect_elem, "width").text = f"{2.0 * extent.y:.4f}"
            except Exception as e:
                 print(f"Warning: Could not get bounding box for vehicle {numeric_id}. Using defaults. Error: {e}")
                 shape_elem = ET.SubElement(obs_elem, "shape")
                 rect_elem = ET.SubElement(shape_elem, "rectangle")
                 ET.SubElement(rect_elem, "length").text = "4.5"
                 ET.SubElement(rect_elem, "width").text = "2.0"

            initial_state_data = states[0]
            init_state_elem = ET.SubElement(obs_elem, "initialState")
            time_e = ET.SubElement(init_state_elem, "time")
            ET.SubElement(time_e, "exact").text = "0"
            pos_x, pos_y, _ = initial_state_data['position']
            pos_e = ET.SubElement(init_state_elem, "position")
            point_e = ET.SubElement(pos_e, "point")
            ET.SubElement(point_e, "x").text = f"{pos_x:.4f}"
            ET.SubElement(point_e, "y").text = f"{pos_y:.4f}"
            yaw_deg = initial_state_data['orientation_deg']
            yaw_rad = math.radians(yaw_deg)
            orient_e = ET.SubElement(init_state_elem, "orientation")
            ET.SubElement(orient_e, "exact").text = f"{yaw_rad:.4f}"
            vel_x, vel_y, vel_z = initial_state_data['velocity_mps']
            speed = math.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
            vel_e = ET.SubElement(init_state_elem, "velocity")
            ET.SubElement(vel_e, "exact").text = f"{speed:.4f}"
            accel = initial_state_data.get('acceleration_mps2', 0.0)
            accel_e = ET.SubElement(init_state_elem, "acceleration")
            ET.SubElement(accel_e, "exact").text = f"{accel:.4f}"

            traj_elem = ET.SubElement(obs_elem, "trajectory")
            start_time_abs = initial_state_data['timestamp']
            all_timestamps = np.array([s['timestamp'] for s in states])

            target_relative_time = time_step_output
            output_time_integer = 1
            last_original_index = 0 # Start search from the first state *after* initial_state (states[0])

            for current_original_idx in range(1, len(states)):
                current_state_data = states[current_original_idx]

                pass # The original while loop for trajectory states seems correct.



            # --- Trajectory (States at integer time steps representing time_step_output intervals) ---
            # start_time_abs is from states[0]
            # all_timestamps is from all states[0...N-1]
            

            target_relative_time = time_step_output # First target is time_step_output after initial state
            output_time_integer = 1             # Corresponds to <state time="1">
            # last_original_index should be 0, as states[0] is initial, we are looking for states for time > 0
            current_search_start_idx = 0 # Start search from the beginning of 'states' list for each target time.

            
            processed_state_timestamps_for_traj = set() # To avoid using the same raw state for multiple XML states

            sim_timestamps_array = np.array([s['timestamp'] for s in states])

            for i_output_step in range(1, int((states[-1]['timestamp'] - start_time_abs) / time_step_output) + 2): # Max possible steps
                target_abs_time_for_step = start_time_abs + i_output_step * time_step_output
                

            current_search_idx_in_states = 0 # Index into this vehicle's 'states' list
            max_output_steps_for_vehicle = int((states[-1]['timestamp'] - start_time_abs) / time_step_output) + 5 # Safety margin

            for _output_step_counter_unused in range(max_output_steps_for_vehicle): # Iterate for desired output steps
                target_absolute_time = start_time_abs + target_relative_time

                # Find the best match in the remainder of the 'states' list
                best_match_overall_idx = -1
                min_time_diff_to_target = float('inf')

                for k_state_idx in range(current_search_idx_in_states, len(states)):
                    time_diff = abs(states[k_state_idx]['timestamp'] - target_absolute_time)
                    if time_diff < min_time_diff_to_target:
                        min_time_diff_to_target = time_diff
                        best_match_overall_idx = k_state_idx
                    # Optimization: if we start moving away from target, and we had a good match
                    if states[k_state_idx]['timestamp'] > target_absolute_time and min_time_diff_to_target < time_step_output / 2.0:
                        break 
                
                if best_match_overall_idx == -1 or min_time_diff_to_target > time_step_output * 0.6 : # If no good match or too far
                    # No suitable state found for this output_time_integer, or we are past available data
                    # Check if we should advance target_relative_time anyway if we skipped a step
                    if target_absolute_time > states[-1]['timestamp'] + time_step_output: # well past end of data
                         break
                    target_relative_time += time_step_output # Try for next time step

                    break 

            start_time_abs_for_traj = states[0]['timestamp'] # This is time=0 in CR
            all_timestamps_for_traj = np.array([s['timestamp'] for s in states]) # Includes states[0]

            output_time_int = 1 # XML trajectory state time starts at 1
            current_raw_state_search_idx = 0 # Search from the beginning of this vehicle's states list

            while True:
                target_sim_time = start_time_abs_for_traj + (output_time_int * time_step_output)

                if current_raw_state_search_idx >= len(all_timestamps_for_traj):
                    break # No more raw states to search

                remaining_timestamps = all_timestamps_for_traj[current_raw_state_search_idx:]
                if remaining_timestamps.size == 0:
                    break
                
                time_diffs = np.abs(remaining_timestamps - target_sim_time)
                best_match_relative_idx = time_diffs.argmin()
                best_match_absolute_idx = current_raw_state_search_idx + best_match_relative_idx
                
                actual_sim_time_of_match = all_timestamps_for_traj[best_match_absolute_idx]


                if time_diffs[best_match_relative_idx] > time_step_output * 0.55: # Allow a bit over 0.5 for edge cases
                    # No good data point for this `output_time_int`.
                    # Maybe advance `output_time_int` and try again?
                    # Or, if the `target_sim_time` is already past all available data, then break.
                    if target_sim_time > all_timestamps_for_traj[-1] + time_step_output * 0.55 :
                        break # We are looking for a time far beyond any available data
                    
                    output_time_int += 1 # Try for the next XML time step
                    # Don't advance current_raw_state_search_idx yet, the current raw states might be useful for the *next* output_time_int
                    if output_time_int > max_output_steps_for_vehicle : # Safety break
                        break
                    continue


                matched_state_data = states[best_match_absolute_idx]

                # Write this state to XML
                state_elem = ET.SubElement(traj_elem, "state")
                time_e = ET.SubElement(state_elem, "time")
                ET.SubElement(time_e, "exact").text = str(output_time_int)

                pos_x, pos_y, _ = matched_state_data['position']
                pos_e = ET.SubElement(state_elem, "position")
                point_e = ET.SubElement(pos_e, "point")
                ET.SubElement(point_e, "x").text = f"{pos_x:.4f}"
                ET.SubElement(point_e, "y").text = f"{pos_y:.4f}"

                yaw_deg = matched_state_data['orientation_deg']
                yaw_rad = math.radians(yaw_deg)
                orient_e = ET.SubElement(state_elem, "orientation")
                ET.SubElement(orient_e, "exact").text = f"{yaw_rad:.4f}"

                vel_x, vel_y, vel_z = matched_state_data['velocity_mps']
                speed = math.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
                vel_e = ET.SubElement(state_elem, "velocity")
                ET.SubElement(vel_e, "exact").text = f"{speed:.4f}"

                accel = matched_state_data.get('acceleration_mps2', 0.0)
                accel_e = ET.SubElement(state_elem, "acceleration")
                ET.SubElement(accel_e, "exact").text = f"{accel:.4f}"

                # Advance for next XML step and next search
                output_time_int += 1
                current_raw_state_search_idx = best_match_absolute_idx + 1 # Next search must start after this used state

                if output_time_int > max_output_steps_for_vehicle : # Safety break
                     print(f"Warning: Exiting trajectory loop for {vehicle_id_str} due to max output steps.")
                     break


        except Exception as e:
            print(f"Error processing trajectory for vehicle {vehicle_id_str}: {e}")
            import traceback
            traceback.print_exc()

    print(f"Added {obstacle_count} dynamic obstacles to XML.")

    try:
        rough_string = ET.tostring(root, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        pretty_xml_as_string = reparsed.toprettyxml(indent="  ")
        with open(output_filename, "w", encoding='utf-8') as f:
            f.write(pretty_xml_as_string)
        print(f"Successfully wrote combined XML to {output_filename}")
    except Exception as e:
        print(f"Error writing XML file: {e}")
        try:
            tree = ET.ElementTree(root)
            tree.write(output_filename, encoding='utf-8', xml_declaration=True)
            print(f"Successfully wrote combined XML to {output_filename} (no pretty print).")
        except Exception as e2:
             print(f"Could not write XML file even without pretty printing: {e2}")


# --- Function to Save Map as PNG ---
def save_map_as_png(all_left_boundaries, all_right_boundaries, output_filename="map_visualization.png", dpi=300):
    """ Plots lane boundaries and saves as PNG. """
    print(f"Generating map visualization and saving to {output_filename}...")
    if not all_left_boundaries and not all_right_boundaries:
        print("No boundary data to plot.")
        return
    try:
        fig, ax = plt.subplots(figsize=(15, 15))
        min_x, max_x, min_y, max_y = float('inf'), float('-inf'), float('inf'), float('-inf')
        all_x = []
        all_y = []
        for segment_points in all_left_boundaries + all_right_boundaries:
            if segment_points:
                all_x.extend([p[0] for p in segment_points])
                all_y.extend([p[1] for p in segment_points])
        if not all_x or not all_y:
             print("Warning: No valid boundary points found to plot.")
             plt.close(fig)
             return
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        for segment_points in all_left_boundaries:
            if not segment_points: continue
            x_coords = [p[0] for p in segment_points]
            y_coords = [p[1] for p in segment_points]
            ax.plot(x_coords, y_coords, color='darkgray', linewidth=0.7)
        for segment_points in all_right_boundaries:
             if not segment_points: continue
             x_coords = [p[0] for p in segment_points]
             y_coords = [p[1] for p in segment_points]
             ax.plot(x_coords, y_coords, color='darkgray', linewidth=0.7)
        ax.set_xlabel("X coordinate (m)")
        ax.set_ylabel("Y coordinate (m)")
        ax.set_title("CARLA Map Lane Boundaries")
        ax.set_aspect('equal', adjustable='box')
        padding_x = (max_x - min_x) * 0.05 if max_x > min_x else 5
        padding_y = (max_y - min_y) * 0.05 if max_y > min_y else 5
        ax.set_xlim(min_x - padding_x, max_x + padding_x)
        ax.set_ylim(min_y - padding_y, max_y + padding_y)
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(output_filename, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"Map visualization saved successfully to {output_filename}")
    except ImportError:
        print("Error: Matplotlib is required to save the map as PNG. Install with: pip install matplotlib")
    except Exception as e:
        print(f"Error generating or saving map PNG: {e}")
        if 'fig' in locals(): plt.close(fig)


# --- Carla Connection ---
def connect_to_carla(host='localhost', port=2000, retries=5, delay=5.0):
    """ Connects to CARLA server with retries. """
    for attempt in range(retries):
        try:
            client = carla.Client(host, port)
            client.set_timeout(20.0)
            world = client.get_world()
            map_name = world.get_map().name
            print(f"Connected to CARLA server. Map: {map_name}")
            return client, world
        except Exception as e:
            print(f"Connection attempt {attempt + 1}/{retries} failed: {e}")
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Maximum connection attempts reached.")
    raise RuntimeError("Failed to connect to CARLA server after multiple attempts.")


if __name__ == "__main__":
    NUM_SIMULATIONS = 1500
    AVAILABLE_MAPS = ["Town01", "Town02", "Town03", "Town04", "Town05"]
    client = None
    original_settings = None
    error_occurred = False

    # --- MODIFICATION: Define radius for local context ---
    LOCAL_CONTEXT_RADIUS = 75.0

    for sim_idx in range(NUM_SIMULATIONS):
        print(f"\n=== Starting Simulation {sim_idx + 1}/{NUM_SIMULATIONS} ===")
        world = None
        vehicles_actors_master_list = [] # --- MODIFICATION: To store all spawned actors for cleanup and XML metadata ---
        ego_vehicle = None # --- NEW: To store the selected ego vehicle ---
        left_boundaries = []
        right_boundaries = []
        processed_keys = set()
        trajectory_data = {}
        XML_TRAJECTORY_TIME_STEP = 0.2
        SIMULATION_DELTA_SECONDS = 0.1 # CARLA simulation step time

        try:
            client, world = connect_to_carla()
            selected_map = random.choice(AVAILABLE_MAPS)
            print(f"Loading map: {selected_map}")
            current_map_name = world.get_map().name.split('/')[-1]
            if current_map_name != selected_map: # Only load if different
                client.load_world(selected_map)
                time.sleep(5.0) # Increased sleep for map loading
            else:
                print(f"Map {selected_map} is already loaded.")
                # If map is already loaded, clean up actors from previous sim if any lingered
                # This is a safety measure, normal cleanup should handle it.
                actor_list = world.get_actors()
                vehicles_to_destroy = [actor for actor in actor_list.filter('vehicle.*')]
                if vehicles_to_destroy:
                    print(f"Cleaning up {len(vehicles_to_destroy)} vehicles from potentially previous run on same map.")
                    client.apply_batch_sync([carla.command.DestroyActor(v.id) for v in vehicles_to_destroy], True)
                    time.sleep(1.0)

            carla_map = world.get_map()

            print("\n--- 1. Extracting Map Geometry ---")
            left_boundaries, right_boundaries, processed_keys = extract_map_data(carla_map, segment_resolution=2.0)

            print("\n--- 2. Running Simulation ---")
            original_settings = world.get_settings()
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = SIMULATION_DELTA_SECONDS
            world.apply_settings(settings)

            traffic_manager_port = 8000 # Default TM port
            traffic_manager = client.get_trafficmanager(traffic_manager_port)
            traffic_manager.set_synchronous_mode(True)
            # traffic_manager.set_global_distance_to_leading_vehicle(3.0) # Will be set per vehicle
            traffic_manager.set_hybrid_physics_mode(True)
            traffic_manager.set_hybrid_physics_radius(70.0) # TM's own radius for detailed physics
            print(f"Synchronous mode enabled ({1/SIMULATION_DELTA_SECONDS:.1f} Hz). TM hybrid physics enabled.")

            blueprint_library = world.get_blueprint_library()
            vehicle_bps = blueprint_library.filter('vehicle.*')
            spawn_points = carla_map.get_spawn_points()
            available_spawn_points = list(spawn_points) if spawn_points else []
            random.shuffle(available_spawn_points)

            # --- MODIFICATION: Higher density target ---
            num_vehicles_target = random.randint(40, 80) # Target 40-80 vehicles
            num_vehicles_to_spawn = min(num_vehicles_target, len(available_spawn_points))
            print(f"Attempting to spawn {num_vehicles_to_spawn} vehicles (target: {num_vehicles_target}, available_spawn_points: {len(available_spawn_points)})")


            for i in range(num_vehicles_to_spawn):
                if not available_spawn_points: break
                bp = random.choice(vehicle_bps)
                if bp.has_attribute('color'): bp.set_attribute('color', random.choice(bp.get_attribute('color').recommended_values))
                if bp.has_attribute('driver_id'): bp.set_attribute('driver_id', random.choice(bp.get_attribute('driver_id').recommended_values))

                spawn_point = available_spawn_points.pop(0)
                vehicle = world.try_spawn_actor(bp, spawn_point)
                if vehicle is not None:
                                    vehicle.set_autopilot(True, traffic_manager.get_port()) # Get TM port correctly
                                    
   
                                    traffic_manager.auto_lane_change(vehicle, True) # Enable auto lane change for this vehicle
                                    
                                    # Set desired distance to leading vehicle for this specific vehicle
                                    traffic_manager.distance_to_leading_vehicle(vehicle, random.uniform(2.5, 5.0))
                                    


                                    desired_speed_modifier_percent = random.uniform(-30.0, 10.0)
                                    traffic_manager.vehicle_percentage_speed_difference(vehicle, desired_speed_modifier_percent)
                                    


                                    vehicles_actors_master_list.append(vehicle)

            print(f"Successfully spawned {len(vehicles_actors_master_list)} vehicles.")


            # --- NEW: Select Ego Vehicle ---
            if not vehicles_actors_master_list:
                print("No vehicles were successfully spawned. Skipping simulation scenario.")
                simulation_duration_sec = 0 # No simulation if no vehicles
            else:
                ego_vehicle = random.choice(vehicles_actors_master_list)
                print(f"Selected Ego Vehicle: {ego_vehicle.id} ({ego_vehicle.type_id}) at {ego_vehicle.get_location()}")
                # Ensure ego's trajectory data list is initialized
                trajectory_data[f"vehicle_{ego_vehicle.id}"] = []
                simulation_duration_sec = 30 # Duration of the scenario in seconds

            # Brief pause for TM to initialize vehicles
            if vehicles_actors_master_list:
                 world.tick() # Initial tick to let TM take control
                 print(f"Waiting a moment for {len(vehicles_actors_master_list)} vehicles to settle...")
                 for _ in range(int(0.5 / SIMULATION_DELTA_SECONDS)): # Wait 0.5 seconds
                     world.tick()
                 print("Starting trajectory collection.")


            simulation_steps = int(simulation_duration_sec / SIMULATION_DELTA_SECONDS)
            last_vehicle_state = {} # For acceleration calculation {vehicle_id: {timestamp, velocity_vec}}
            
            # This list will hold actors that are currently alive and managed in the loop
            current_sim_actors = list(vehicles_actors_master_list)


            print(f"Simulating for {simulation_duration_sec} seconds ({simulation_steps} steps)...")
            actual_steps_run = 0
            for step_idx in range(simulation_steps):
                actual_steps_run = step_idx + 1
                snapshot = world.get_snapshot()
                timestamp = snapshot.timestamp.elapsed_seconds

                # --- NEW: Ego-centric filtering ---
                ego_location = None
                if ego_vehicle and ego_vehicle.is_alive:
                    ego_location = ego_vehicle.get_location()
                else:
                    if ego_vehicle: print(f"Ego vehicle {ego_vehicle.id} is no longer alive. Stopping simulation early.")
                    else: print("No ego vehicle was selected/valid. Stopping simulation early.")
                    break # Exit simulation_steps loop

                active_actors_this_tick = [] # Actors that are alive before this tick's processing
                
                for actor in current_sim_actors:
                    if not actor.is_alive:
                        if actor.id == ego_vehicle.id: # Should have been caught above, but double check
                            print(f"Ego vehicle {actor.id} found dead during actor iteration. This is unexpected here.")
                            ego_vehicle = None # Mark ego as gone
                        continue # Skip dead actors

                    active_actors_this_tick.append(actor) # Keep track of actors alive at start of this tick

                    # Determine if this actor's data should be recorded
                    record_this_actor_data = False
                    if actor.id == ego_vehicle.id:
                        record_this_actor_data = True
                    else: # It's a context agent
                        if ego_location.distance(actor.get_location()) <= LOCAL_CONTEXT_RADIUS:
                            record_this_actor_data = True
                    
                    if record_this_actor_data:
                        vehicle_key = f"vehicle_{actor.id}"
                        if vehicle_key not in trajectory_data: # First time this context agent is in range
                            trajectory_data[vehicle_key] = []
                        
                        try:
                            transform = actor.get_transform()
                            velocity_vec = actor.get_velocity() # carla.Vector3D
                            current_speed = velocity_vec.length() # float

                            acceleration = 0.0
                            prev_state = last_vehicle_state.get(actor.id)
                            if prev_state:
                                delta_time = timestamp - prev_state['timestamp']
                                if delta_time > 1e-6: # Avoid division by zero
                                    prev_speed = prev_state['velocity_vec'].length()
                                    acceleration = (current_speed - prev_speed) / delta_time
                            
                            trajectory_data[vehicle_key].append({
                                'timestamp': timestamp,
                                'position': (transform.location.x, transform.location.y, transform.location.z),
                                'orientation_deg': transform.rotation.yaw,
                                'velocity_mps': (velocity_vec.x, velocity_vec.y, velocity_vec.z),
                                'acceleration_mps2': acceleration
                            })
                            last_vehicle_state[actor.id] = {
                                'timestamp': timestamp,
                                'velocity_vec': velocity_vec # Store carla.Vector3D
                            }
                        except Exception as e:
                            print(f"Warning: Error getting data for {vehicle_key} at step {step_idx}: {e}")
                            if actor.id in last_vehicle_state: del last_vehicle_state[actor.id]
                
                world.tick() # Advance simulation state

                current_sim_actors = active_actors_this_tick # Update list for next iteration
                if not current_sim_actors: # Or specifically, if ego is not in active_actors_this_tick and was expected
                    print("No active actors remaining (or ego is gone). Stopping simulation early.")
                    break
                
                # Post-tick check for ego specifically, as physics might have affected it
                if ego_vehicle and not ego_vehicle.is_alive:
                    print(f"Ego vehicle {ego_vehicle.id} died during tick {step_idx}. Stopping simulation.")
                    break

            print(f"Simulation finished after {actual_steps_run} steps.")

            print("\n--- 3. Saving Combined Output ---")
            output_dir = f"simulation_outputs_+108{selected_map}"
            os.makedirs(output_dir, exist_ok=True)
            output_xml_filename = f"sim_output_{selected_map}_idx{sim_idx + 1}.xml"
            output_xml_path = os.path.join(output_dir, output_xml_filename)

            # Pass the original full list of spawned vehicles for metadata (like bounding boxes)
            write_combined_xml(carla_map, left_boundaries, right_boundaries, processed_keys,
                               vehicles_actors_master_list, trajectory_data,
                               output_filename=output_xml_path,
                               time_step_output=XML_TRAJECTORY_TIME_STEP)

            # print("\n--- 4. Saving Map Visualization ---")
            # output_png_filename = f"map_viz_{selected_map}_idx{sim_idx + 1}.png"
            # output_png_path = os.path.join(output_dir, output_png_filename)
            # save_map_as_png(left_boundaries, right_boundaries, output_filename=output_png_path)


        except RuntimeError as e:
            print(f"Runtime Error in simulation {sim_idx + 1} ({selected_map if 'selected_map' in locals() else 'UnknownMap'}): {e}")
            error_occurred = True
        except Exception as e:
            print(f"An unexpected error occurred in simulation {sim_idx + 1} ({selected_map if 'selected_map' in locals() else 'UnknownMap'}): {e}")
            import traceback
            traceback.print_exc()
            error_occurred = True
        finally:
            print("\n--- Cleaning up ---")
            if world is not None:
                if original_settings is not None:
                    try:
                        print("Restoring original world settings...")
                        world.apply_settings(original_settings)
                    except Exception as e_clean_settings:
                        print(f"Error during settings restore: {e_clean_settings}")
                
                # Always try to disable sync mode for TM if it was used
                if 'traffic_manager' in locals() and traffic_manager:
                    try:
                        traffic_manager.set_synchronous_mode(False)
                        print("Traffic Manager synchronous mode disabled.")
                    except Exception as e_tm_sync:
                        print(f"Error disabling TM sync mode: {e_tm_sync}")

            if client is not None and 'vehicles_actors_master_list' in locals() and vehicles_actors_master_list:
                print(f"Destroying {len(vehicles_actors_master_list)} spawned vehicle actors...")
                # Check aliveness before attempting to destroy
                actors_to_destroy_ids = []
                if world: # Check if world object is valid
                    all_actors_in_world = world.get_actors()
                    world_actor_ids = {act.id for act in all_actors_in_world}
                    for v_actor in vehicles_actors_master_list:
                        if v_actor.is_alive and v_actor.id in world_actor_ids : # Check if CARLA still knows about it
                           actors_to_destroy_ids.append(v_actor.id)
                        # else:
                        #    print(f"Actor {v_actor.id} reported as not alive or not in world, skipping destroy command.")
                else: # Fallback if world object is None, try to destroy based on stored list
                    actors_to_destroy_ids = [v.id for v in vehicles_actors_master_list]


                if actors_to_destroy_ids:
                    print(f"Attempting to destroy {len(actors_to_destroy_ids)} actors.")
                    client.apply_batch_sync([carla.command.DestroyActor(actor_id) for actor_id in actors_to_destroy_ids], True) # Synchronous destruction
                    time.sleep(0.5) # Give server a moment
                else:
                    print("No live actors from this simulation instance found to destroy.")
            
            vehicles_actors_master_list = [] # Clear for next simulation
            ego_vehicle = None
            # Reset original_settings to ensure it's fetched fresh if next sim fails early
            original_settings = None


            print(f"Cleanup finished for simulation {sim_idx + 1}.")
            # A small delay before starting the next simulation, especially if maps are reloaded.
            # time.sleep(1.0) 

    print("\n=== All Simulations Completed ===")
    sys.exit(1 if error_occurred else 0)
