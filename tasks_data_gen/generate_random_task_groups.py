import json, random

task_types=[
    #For rover-related tasks
    "ROVER",
    #For checking vitals
    "VITALS",
    #For repair tasks
    "REPAIR",
    #For navigation/movement tasks
    "NAVIGATION",
    #For geological sampling tasks
    "GEOSAMPLING",
    #For emergency tasks
    "EMERGENCY",
    #For emergency subtasks
    "SUB_EMERGENCY",
    #For normal subtasks
    "SUB_NORMAL"
]
potential_tasks = [
  {
    "task": "Navigate to designated waypoint alpha",
    "task_type": "NAVIGATION",
    "subtasks": ["Verify current coordinates", "Plot course avoiding obstacles", "Engage autonomous navigation", "Confirm arrival at waypoint"]
  },

  {
    "task": "Avoid marked danger zone in sector 3", 
    "task_type": "NAVIGATION",
    "subtasks": ["Access hazard map overlay", "Calculate detour route", "Adjust thrusters for course correction", "Monitor proximity alerts"]
  },

  {
    "task": "Place navigation waypoint at coordinates",
    "task_type": "NAVIGATION", 
    "subtasks": ["Input GPS coordinates", "Validate coordinate accuracy", "Set temporary marker", "Update mission log with waypoint"]
  },

  {
    "task": "Follow exploration path beta",
    "task_type": "NAVIGATION",
    "subtasks": ["Activate pre-programmed route", "Maintain 5km/h traversal speed", "Check progress at 500m intervals", "Adjust for terrain irregularities"]

  },
  {
    "task": "Calculate optimal route to base",
    "task_type": "NAVIGATION",
    "subtasks": ["Analyze terrain elevation data", "Factor in remaining battery life", "Select shortest safe path", "Confirm route with team"]
  },
  {
    "task": "Mark hazardous terrain in quadrant 2",
    "task_type": "NAVIGATION",
    "subtasks": ["Perform LIDAR surface scan", "Tag unstable areas on shared map", "Broadcast warning to team", "Update mission control"]
  },
  {
    "task": "Navigate back to lunar lander",
    "task_type": "NAVIGATION",
    "subtasks": ["Verify lander beacon signal", "Initiate return sequence", "Monitor navigation system status", "Perform final approach checks"]
  },
  {
    "task": "Locate emergency shelter point",
    "task_type": "EMERGENCY",
    "subtasks": ["Access shelter database", "Cross-reference with current position", "Set direct path to nearest shelter", "Confirm structural integrity"]
  },
  {
    "task": "Track distance traveled so far",
    "task_type": "NAVIGATION",
    "subtasks": ["Activate odometer tracking", "Record hourly progress", "Compare actual vs planned distance", "Update EVA log"]
  },
  {
    "task": "Monitor elevation changes",
    "task_type": "NAVIGATION",
    "subtasks": ["Deploy laser altimeter", "Log elevation every 100 meters", "Watch for sudden drops/inclines", "Report significant changes"]
  },
  {
    "task": "Perform UIA egress procedure",
    "task_type": "ROVER",
    "subtasks": ["Confirm suit pressure integrity", "Depressurize airlock", "Secure tethers and tools", "Execute three-step exit protocol"]
  },
  {
    "task": "Monitor spacesuit vital signs",
    "task_type": "VITALS",
    "subtasks": ["Check oxygen partial pressure", "Review CO2 scrubber status", "Monitor core body temperature", "Alert for any parameter breaches"]
  },
  {
    "task": "Issue rover movement commands",
    "task_type": "ROVER",
    "subtasks": ["Input destination coordinates", "Confirm rover system status", "Execute movement authorization", "Track progress via cameras"]
  },
  {
    "task": "Activate primary life support",
    "task_type": "VITALS",
    "subtasks": ["Power up oxygen generation", "Verify CO2 removal cycling", "Check water cooling flow", "Confirm stable pressure readings"]
  },
  {
    "task": "Check oxygen level status",
    "task_type": "VITALS",
    "subtasks": ["Review primary tank levels", "Calculate remaining duration", "Inspect for leaks or damage", "Report to mission control"]
  },
  {
    "task": "Verify communication systems",
    "task_type": "REPAIR",
    "subtasks": ["Test UHF/VHF channels", "Check signal strength meter", "Verify backup transmitter", "Establish test link with Earth"]
  },
  {
    "task": "Calibrate sample analysis tools",
    "task_type": "GEOSAMPLING",
    "subtasks": ["Run diagnostic sequence", "Adjust spectrometer alignment", "Test with control samples", "Validate calibration accuracy"]
  },
  {
    "task": "Deploy solar panel array",
    "task_type": "REPAIR",
    "subtasks": ["Release locking mechanisms", "Extend panel segments", "Confirm sun alignment angle", "Monitor power generation levels"]
  },
  {
    "task": "Activate backup systems",
    "task_type": "EMERGENCY",
    "subtasks": ["Switch to secondary power", "Test failover mechanisms", "Monitor primary system status", "Update maintenance log"]
  },
  {
    "task": "Test radiation shielding",
    "task_type": "VITALS",
    "subtasks": ["Deploy neutron detectors", "Measure cosmic ray exposure", "Check shield layer integrity", "Adjust mobile protection units"]
  },
  {
    "task": "Collect geological samples",
    "task_type": "GEOSAMPLING",
    "subtasks": ["Identify collection site", "Use rock hammer and scoop", "Store in sterile containers", "Perform initial spectrometry"]
  },
  {
    "task": "Deploy sample marker beacon",
    "task_type": "GEOSAMPLING",
    "subtasks": ["Position near sample site", "Activate flashing LED", "Test radio frequency ID", "Log GPS coordinates"]
  },
  {
    "task": "Log current sample location",
    "task_type": "GEOSAMPLING",
    "subtasks": ["Record lunar coordinates", "Note surrounding landmarks", "Capture 360-degree imagery", "Update field database"]
  },
  {
    "task": "Take sample site photos",
    "task_type": "GEOSAMPLING",
    "subtasks": ["Set camera to macro mode", "Capture stratigraphic layers", "Include scale reference", "Tag with metadata"]
  },
  {
    "task": "Measure sample dimensions",
    "task_type": "GEOSAMPLING",
    "subtasks": ["Use laser calipers", "Record length/width/depth", "Note geometric shape", "Compare to nearby specimens"]
  },
  {
    "task": "Record sample properties",
    "task_type": "GEOSAMPLING",
    "subtasks": ["Test magnetic response", "Note color and texture", "Measure specific gravity", "Document crystalline structures"]
  },
  {
    "task": "Package sample for transport",
    "task_type": "GEOSAMPLING",
    "subtasks": ["Seal in argon-filled container", "Apply anti-contamination coating", "Secure in sample rack", "Verify quarantine protocols"]
  },
  {
    "task": "Label specimen container",
    "task_type": "GEOSAMPLING",
    "subtasks": ["Use radiation-resistant tags", "Include collection time/date", "Note suspected mineral content", "Apply hazard warnings"]
  },
  {
    "task": "Document sampling conditions",
    "task_type": "GEOSAMPLING",
    "subtasks": ["Record ambient temperature", "Note solar radiation levels", "Document surface hardness", "Log tool usage statistics"]
  },
  {
    "task": "Update sample inventory",
    "task_type": "GEOSAMPLING",
    "subtasks": ["Scan container QR codes", "Reconcile field counts", "Archive in central database", "Transmit update to orbiter"]
  },
  {
    "task": "Contact mission control",
    "task_type": "EMERGENCY",
    "subtasks": ["Establish satellite link", "Send status ping", "Await confirmation handshake", "Log communication timestamp"]
  },
  {
    "task": "Send message to crew",
    "task_type": "NAVIGATION",
    "subtasks": ["Compose text transmission", "Select encrypted channel", "Verify recipient list", "Get transmission confirmation"]
  },
  {
    "task": "Report equipment status",
    "task_type": "REPAIR",
    "subtasks": ["Generate system report", "Highlight anomalies", "Attach diagnostic logs", "Send priority message"]
  },
  {
    "task": "Request technical support",
    "task_type": "REPAIR",
    "subtasks": ["Describe issue in detail", "Include error codes", "Specify urgency level", "Attach relevant sensor data"]
  },
  {
    "task": "Coordinate team positions",
    "task_type": "NAVIGATION",
    "subtasks": ["Request location updates", "Plot on shared map", "Identify regrouping points", "Broadcast movement plan"]
  },
  {
    "task": "Share scientific data",
    "task_type": "GEOSAMPLING",
    "subtasks": ["Compress data files", "Verify checksums", "Transmit via high-gain antenna", "Confirm successful receipt"]
  },
  {
    "task": "Report emergency situation",
    "task_type": "EMERGENCY",
    "subtasks": ["Activate emergency channel", "Broadcast SOS signal", "Follow contingency protocols", "Maintain open comm line"]
  },
  {
    "task": "Confirm mission objectives",
    "task_type": "NAVIGATION",
    "subtasks": ["Review task checklist", "Verify completion status", "Clarify ambiguous items", "Obtain formal confirmation"]
  },
  {
    "task": "Document procedure completion",
    "task_type": "NAVIGATION",
    "subtasks": ["Record completion time", "Capture verification photos", "Obtain team signatures", "Archive in mission log"]
  },
  {
    "task": "Relay environmental readings",
    "task_type": "VITALS",
    "subtasks": ["Collect sensor array data", "Compile into standard format", "Transmit hourly update", "Backup to local storage"]
  },
  { 
    "task": "Execute low-orbit stationkeeping maneuver",
    "task_type": "NAVIGATION",
    "subtasks": ["Calculate thruster burn duration", "Monitor orbital decay rate", "Adjust altitude via RCS", "Confirm position relative to target"] 
  },
  { 
    "task": "Plot course through asteroid field",
    "task_type": "NAVIGATION", 
    "subtasks": ["Map asteroid trajectories", "Identify safe corridors", "Program evasive maneuvers", "Activate forward collision sensors"] 
  },
  { 
    "task": "Calibrate lunar south pole navigation beacons",
    "task_type": "NAVIGATION",
    "subtasks": ["Locate beacon array", "Test signal strength", "Adjust broadcast frequencies", "Validate polar coordinate alignment"] 
  },
  { 
    "task": "Initiate emergency ascent from surface",
    "task_type": "EMERGENCY",
    "subtasks": ["Override auto-navigation", "Prioritize vertical thrust", "Monitor fuel consumption rates", "Align with orbital rescue vector"] 
  },
  { 
    "task": "Optimize trajectory for fuel efficiency",
    "task_type": "NAVIGATION",
    "subtasks": ["Analyze gravitational slingshot options", "Simulate multiple routes", "Select minimal delta-V path", "Lock in final trajectory"] 
  },
  { 
    "task": "Coordinate lunar rover convoy",
    "task_type": "ROVER",
    "subtasks": ["Establish lead/follow protocols", "Sync navigation systems", "Maintain safe separation distances", "Implement swarm intelligence routing"] 
  },
  { 
    "task": "Perform orbital insertion burn",
    "task_type": "NAVIGATION",
    "subtasks": ["Align spacecraft orientation", "Verify engine gimbal range", "Execute timed thrust sequence", "Confirm orbital parameters post-burn"] 
  },
  { 
    "task": "Navigate using stellar reference",
    "task_type": "NAVIGATION",
    "subtasks": ["Acquire guide stars", "Calculate celestial coordinates", "Cross-check with inertial measurement", "Update dead reckoning system"] 
  },
  { 
    "task": "Avoid micrometeoroid swarm",
    "task_type": "EMERGENCY",
    "subtasks": ["Analyze radar tracking data", "Rotate shielded side toward threat", "Prepare for potential impacts", "Document swarm characteristics"] 
  },
  { 
    "task": "Dock with rotating space station",
    "task_type": "NAVIGATION",
    "subtasks": ["Match rotation axis", "Synchronize angular velocity", "Deploy capture mechanism", "Confirm hard dock seal"] 
  },
  { 
    "task": "Map subsurface lava tubes",
    "task_type": "GEOSAMPLING",
    "subtasks": ["Deploy ground-penetrating radar", "Analyze echo patterns", "Tag potential shelter sites", "Update geological database"] 
  },
  { 
    "task": "Perform emergency course reversal",
    "task_type": "EMERGENCY",
    "subtasks": ["Calculate flip maneuver", "Redirect main engines", "Stabilize during thrust", "Verify new trajectory"] 
  },
  { 
    "task": "Navigate during solar storm",
    "task_type": "EMERGENCY",
    "subtasks": ["Activate radiation hardening", "Switch to inertial navigation", "Monitor compass reliability", "Report positional drift"] 
  },
  { 
    "task": "Program autonomous exploration loop",
    "task_type": "ROVER",
    "subtasks": ["Set discovery priorities", "Define safety constraints", "Upload terrain recognition AI", "Enable self-correction protocols"] 
  },
  { 
    "task": "Align for planetary flyby",
    "task_type": "NAVIGATION",
    "subtasks": ["Calculate gravity assist vector", "Adjust approach angle", "Monitor atmospheric buffer zone", "Capture science data during pass"] 
  },
  { 
    "task": "Test emergency airlock override",
    "task_type": "EMERGENCY",
    "subtasks": ["Engage manual pressure release", "Verify redundant seals", "Check emergency lighting", "Simulate rapid egress"] 
  },
  { 
    "task": "Diagnose power grid fluctuations",
    "task_type": "REPAIR",
    "subtasks": ["Isolate faulty modules", "Test capacitor banks", "Balance load distribution", "Implement surge protection"] 
  },
  { 
    "task": "Repair robotic arm joint",
    "task_type": "REPAIR",
    "subtasks": ["Lock arm in safe position", "Replace servo motors", "Calibrate range of motion", "Test payload handling"] 
  },
  { 
    "task": "Reconfigure life support for EVA",
    "task_type": "VITALS",
    "subtasks": ["Divert oxygen reserves", "Adjust CO₂ scrubbing rate", "Monitor suit umbilical pressure", "Sync with airlock cycling"] 
  },
  { 
    "task": "Test emergency water reclamation",
    "task_type": "EMERGENCY",
    "subtasks": ["Activate backup filters", "Process simulated sweat", "Check bacterial levels", "Verify potable output"] 
  },
  { 
    "task": "Charge battery arrays",
    "task_type": "REPAIR",
    "subtasks": ["Orient solar panels", "Monitor charge curves", "Balance cell voltages", "Prevent thermal runaway"] 
  },
  { 
    "task": "Inspect heat shield integrity",
    "task_type": "REPAIR",
    "subtasks": ["Perform thermal camera scan", "Check ablative material thickness", "Test micrometeoroid patches", "Simulate re-entry stresses"] 
  },
  { 
    "task": "Maintain hydroponic oxygen farm",
    "task_type": "VITALS",
    "subtasks": ["Check nutrient levels", "Prune plant growth", "Adjust LED light cycles", "Monitor O₂ output purity"] 
  },
  { 
    "task": "Test emergency habitat inflation",
    "task_type": "EMERGENCY",
    "subtasks": ["Clear deployment area", "Verify gas cartridge pressure", "Monitor unfolding sequence", "Check structural rigidity"] 
  },
  { 
    "task": "Service spacesuit cooling garment",
    "task_type": "REPAIR",
    "subtasks": ["Flush water lines", "Check for leaks", "Test temperature regulation", "Verify moisture wicking"] 
  },
  { 
    "task": "Deploy seismic monitoring network",
    "task_type": "GEOSAMPLING",
    "subtasks": ["Plant sensor spikes", "Test data transmission", "Geolocate each unit", "Sync detection thresholds"] 
  },
  { 
    "task": "Test emergency habitat separation",
    "task_type": "EMERGENCY",
    "subtasks": ["Verify explosive bolts", "Test thruster ignition", "Simulate crew transfer", "Check independent life support"] 
  },
  { 
    "task": "Calibrate meteorology sensors",
    "task_type": "REPAIR",
    "subtasks": ["Test wind speed accuracy", "Verify pressure readings", "Clean solar radiation detectors", "Align precipitation sensors"] 
  },
  { 
    "task": "Overhaul electrolysis oxygen system",
    "task_type": "REPAIR",
    "subtasks": ["Drain water reservoirs", "Replace proton-exchange membranes", "Test hydrogen venting", "Monitor production rates"] 
  },
  { 
    "task": "Test emergency habitat anchoring",
    "task_type": "EMERGENCY",
    "subtasks": ["Fire ground tethers", "Verify penetration depth", "Monitor stability under load", "Check automatic tensioning"] 
  },
  { 
    "task": "Collect regolith gas samples",
    "task_type": "GEOSAMPLING",
    "subtasks": ["Drive collection probe", "Trap subsurface gases", "Transfer to analysis chamber", "Record outgassing rates"] 
  },
  { 
    "task": "Harvest vacuum-frozen ices",
    "task_type": "GEOSAMPLING",
    "subtasks": ["Use cryogenic scoop", "Prevent sublimation loss", "Store in shielded containers", "Label with depth data"] 
  },
  { 
    "task": "Document impact crater ejecta",
    "task_type": "GEOSAMPLING",
    "subtasks": ["Map dispersion patterns", "Collect shock-metamorphosed rocks", "Measure penetration depth", "Classify impactor type"] 
  },
  { 
    "task": "Sample ancient lava flows",
    "task_type": "GEOSAMPLING",
    "subtasks": ["Identify flow margins", "Chip pristine basalt", "Avoid space weathering surfaces", "Note vesicle patterns"] 
  },
  { 
    "task": "Capture atmospheric dust",
    "task_type": "GEOSAMPLING",
    "subtasks": ["Deploy electrostatic panels", "Time collection with dust storms", "Microscopic imaging", "Analyze ionic charge"] 
  },
  { 
    "task": "Extract core samples",
    "task_type": "GEOSAMPLING",
    "subtasks": ["Drive rotary corer", "Preserve stratigraphy", "Seal ends with epoxy", "Log depth markers"] 
  },
  { 
    "task": "Collect solar wind particles",
    "task_type": "GEOSAMPLING",
    "subtasks": ["Deploy foil traps", "Time with solar events", "Retrieve before degradation", "Package in inert gas"] 
  },
  { 
    "task": "Harvest extremophile cultures",
    "task_type": "GEOSAMPLING",
    "subtasks": ["Sterilize collection tools", "Maintain pressure during transfer", "Incubate in growth chambers", "Monitor adaptation"] 
  },
  { 
    "task": "Trap micrometeorite impacts",
    "task_type": "GEOSAMPLING",
    "subtasks": ["Deploy aerogel array", "Calculate exposure time", "Scan for particle tracks", "Extract with micro-tweezers"] 
  },
  { 
    "task": "Sample polar shadow regions",
    "task_type": "GEOSAMPLING",
    "subtasks": ["Use insulated containers", "Prevent sample warming", "Document water ice content", "Test for organics"] 
  },
  { 
    "task": "Collect volcanic glass spherules",
    "task_type": "GEOSAMPLING",
    "subtasks": ["Identify pyroclastic deposits", "Use magnetic rake", "Sort by size distribution", "Package in crush-proof cases"] 
  },
  { 
    "task": "Capture transient lunar phenomena",
    "task_type": "GEOSAMPLING",
    "subtasks": ["Monitor thermal anomalies", "Deploy spectrometer array", "Time-lapse image sequences", "Correlate with seismic data"] 
  },
  { 
    "task": "Sample electrostatic dust layers",
    "task_type": "GEOSAMPLING",
    "subtasks": ["Use non-metallic tools", "Measure charge dissipation", "Store in Faraday containers", "Document levitation effects"] 
  },
  { 
    "task": "Harvest impact melt breccias",
    "task_type": "GEOSAMPLING",
    "subtasks": ["Locate crater central peaks", "Chip glassy matrix", "Avoid terrestrial contamination", "Note shock features"] 
  },
  { 
    "task": "Collect solar flare isotopes",
    "task_type": "GEOSAMPLING",
    "subtasks": ["Deploy neutron detectors", "Time with coronal mass ejections", "Analyze activation products", "Compare to background levels"] 
  },
  { 
    "task": "Establish laser comms link",
    "task_type": "REPAIR",
    "subtasks": ["Align optical terminal", "Compensate for Doppler shift", "Test data burst transmission", "Monitor beam dispersion"] 
  },
  { 
    "task": "Coordinate satellite handoff",
    "task_type": "NAVIGATION",
    "subtasks": ["Predict orbital window", "Test cross-link encryption", "Verify signal continuity", "Log transfer completion"] 
  },
  { 
    "task": "Broadcast emergency recall",
    "task_type": "EMERGENCY",
    "subtasks": ["Override standard protocols", "Use maximum power transmission", "Repeat distress pattern", "Confirm acknowledgments"] 
  },
  { 
    "task": "Encode sensitive mission data",
    "task_type": "REPAIR",
    "subtasks": ["Apply quantum encryption", "Fragment transmission packets", "Use frequency hopping", "Verify checksum validation"] 
  },
  { 
    "task": "Maintain blackout comms silence",
    "task_type": "EMERGENCY",
    "subtasks": ["Power down non-essentials", "Monitor passive sensors only", "Log attempted contacts", "Prepare burst message queue"] 
  },
  { 
    "task": "Test emergency whistle protocol",
    "task_type": "EMERGENCY",
    "subtasks": ["Verify low-power mode", "Encode position in repeating loop", "Test solar charging viability", "Simulate long-duration operation"] 
  },
  { 
    "task": "Relay through orbital data relay",
    "task_type": "NAVIGATION",
    "subtasks": ["Calculate visibility windows", "Compress telemetry data", "Prioritize transmission queues", "Confirm Earth receipt"] 
  },
  { 
    "task": "Establish moon-Earth-Mars network",
    "task_type": "NAVIGATION",
    "subtasks": ["Time signal relays", "Compensate for light delay", "Verify three-way encryption", "Stress-test bandwidth"] 
  },
  { 
    "task": "Test analog backup comms",
    "task_type": "REPAIR",
    "subtasks": ["Deploy wire dipole antenna", "Practice Morse code protocols", "Monitor AM frequencies", "Verify analog signal clarity"] 
  },
  { 
    "task": "Coordinate multi-spectrum blackout",
    "task_type": "EMERGENCY",
    "subtasks": ["Predict solar interference", "Alert all teams in advance", "Prepare stored commands", "Monitor radiation spikes"] 
  },
  { 
    "task": "Implement secure voice channels",
    "task_type": "REPAIR",
    "subtasks": ["Test voice scramblers", "Verify speaker biometrics", "Establish code phrases", "Monitor for eavesdropping"] 
  },
  { 
    "task": "Test emergency flashlight comms",
    "task_type": "EMERGENCY",
    "subtasks": ["Develop light pulse code", "Practice manual operation", "Test maximum visibility range", "Coordinate acknowledgment patterns"] 
  },
  { 
    "task": "Deploy surface repeater network",
    "task_type": "REPAIR",
    "subtasks": ["Space nodes for coverage", "Bury power cables", "Test signal strength map", "Implement self-healing routing"] 
  },
  { 
    "task": "Simulate total comms loss",
    "task_type": "EMERGENCY",
    "subtasks": ["Activate isolation protocols", "Use written message passing", "Test emergency signage", "Practice celestial navigation"] 
  },
  { 
    "task": "Establish meteor scatter link",
    "task_type": "REPAIR",
    "subtasks": ["Calculate ionization trails", "Time transmissions with bursts", "Use high-speed encoding", "Verify packet reception"] 
  }
]


# Create 2D array of 100 groups of random tasks (3-15 tasks per group)
# Generate task groups and write to file
task_groups = []
used_group_ids = set()

for _ in range(100):
    # Generate random size between 3-15, but cap at number of available tasks
    group_size = min(random.randint(3, 15), len(potential_tasks))
    # Use random.sample to get unique tasks
    group = random.sample(potential_tasks, group_size)
    
    # Add random ID to each task
    used_task_ids = set()
    for task in group:
        # Keep generating IDs until we get a unique one
        while True:
            new_id = str(random.randint(10000, 99999))
            if new_id not in used_task_ids:
                task['id'] = new_id
                used_task_ids.add(new_id)
                break
    
    # Generate unique group ID
    while True:
        group_id = str(random.randint(100000, 999999))
        if group_id not in used_group_ids:
            used_group_ids.add(group_id)
            break
            
    # Create group object with ID and tasks
    group_obj = {
        "id": group_id,
        "tasks": group
    }
    task_groups.append(group_obj)

with open('task_groups.json', 'w') as f:
    json.dump(task_groups, f, indent=2)

