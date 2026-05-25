from pathlib import Path
from loaders.run_cadence_context import setup_output_directory

output_dir, _, _ = setup_output_directory()

targets = [f"VMAG{i}" for i in range(11, 20)]
ir_exposures = [1.0, 2.0, 5.0, 10.0, 30.0, 60.0]

template = """# Name of the target star. Its properties are read from the Excel file in the project root.
# If a property is missing there, it will be retrieved from GAIA.
target_name = {target_name}

# Total duration of the simulated observation, in hours.
# The transit is centered in this window, with half of the time before and half after the transit.
# Floating-point values are allowed (e.g. 24 or 24.5).
total_observation_length_h = 11.11

# Exposure time of the Channels in seconds
exposure_NUV_s = 300.0
exposure_VIS_s = 60.0
exposure_NIR_s = {exposure_ir}
"""

for target in targets:
    folder = Path(output_dir) / f"{target}"
    folder.mkdir(parents=True, exist_ok=True)

    for idx, exposure in enumerate(ir_exposures, start=1):
        filename = folder / f"parameter_{idx:03d}.txt"

        content = template.format(
            target_name=target,
            exposure_ir=exposure
        )

        filename.write_text(content, encoding="utf-8")

print(f"Created parameter files in {output_dir}")