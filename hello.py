import sys
sys.path.insert(0, r'C:\Users\Admin\Desktop\My projects\pulse-phisiology-env\engine-build\install\bin')
sys.path.insert(0, r'C:\Users\Admin\Desktop\My projects\pulse-phisiology-env\engine-build\install\python')

from pulse.engine.PulseEngine import PulseEngine
from pulse.cdm.engine import SEDataRequestManager, SEDataRequest
from pulse.cdm.scalars import FrequencyUnit, PressureUnit, VolumeUnit

pulse = PulseEngine()
pulse.log_to_console(True)

data_requests = [
    SEDataRequest.create_physiology_request("HeartRate", unit=FrequencyUnit.Per_min),
    SEDataRequest.create_physiology_request("SystolicArterialPressure", unit=PressureUnit.mmHg),
    SEDataRequest.create_physiology_request("DiastolicArterialPressure", unit=PressureUnit.mmHg),
    SEDataRequest.create_physiology_request("OxygenSaturation"),
    SEDataRequest.create_physiology_request("RespirationRate", unit=FrequencyUnit.Per_min),
    SEDataRequest.create_physiology_request("BloodVolume", unit=VolumeUnit.mL),
]
data_req_mgr = SEDataRequestManager(data_requests)

state_file = r'C:\Users\Admin\Desktop\My projects\pulse-phisiology-env\engine-build\install\bin\states\StandardMale@0s.json'

if not pulse.serialize_from_file(state_file, data_req_mgr):
    print("❌ Failed to load state")
else:
    print("✅ Patient loaded successfully")
    results = pulse.pull_data()
    print(f"\n📊 Initial Vitals:")
    print(f"  Heart Rate:      {results[1]:.1f} bpm")
    print(f"  Systolic BP:     {results[2]:.1f} mmHg")
    print(f"  Diastolic BP:    {results[3]:.1f} mmHg")
    print(f"  SpO2:            {results[4]:.3f}")
    print(f"  Respiration:     {results[5]:.1f} /min")
    print(f"  Blood Volume:    {results[6]:.1f} mL")
    
    print("\n⏱️ Advancing 10 seconds...")
    pulse.advance_time_s(10)
    results = pulse.pull_data()
    print(f"  Heart Rate:      {results[1]:.1f} bpm")
    print(f"  SpO2:            {results[4]:.3f}")
    print(f"  Blood Volume:    {results[6]:.1f} mL")
    print("\n✅ Pulse-ER is ready to build!")