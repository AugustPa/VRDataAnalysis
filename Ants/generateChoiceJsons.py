import json
import os

def generate_json(angle, material1, material2):
    """
    Returns the dictionary corresponding to the JSON spec,
    given an angle and two material strings.
    """
    # The cylinder positions use -angle/2 and +angle/2
    objects = [
        {
            "type": "ScalingCylinder",
            "position": {
                "radius": 60,
                "angle": -angle / 2,
                "height": 0
            },
            "material": material1,
            "scale": {
                "x": 7,
                "y": 100,
                "z": 7
            },
            "visualAngleDegrees": 10
        },
        {
            "type": "ScalingCylinder",
            "position": {
                "radius": 60,
                "angle": angle / 2,
                "height": 0
            },
            "material": material2,
            "scale": {
                "x": 7,
                "y": 100,
                "z": 7
            },
            "visualAngleDegrees": 10
        }
    ]
    
    data = {
        "objects": objects,
        "closedLoopOrientation": True,
        "closedLoopPosition": True,
        "backgroundColor": {
            "r": 0.8,
            "g": 0.8,
            "b": 0.8,
            "a": 1
        }
    }
    
    return data

def main():
    angles = [20, 30, 50, 70, 90, 110, 130, 150, 180]
    
    # We will generate two files per angle:
    # 1) {angle}deg_BlueGreenCylinder_BlueCylinder
    # 2) {angle}deg_BlueCylinder_BlueGreenCylinder
    
    # Create a folder if you want to store the files in a separate directory
    # For simplicity, we'll just dump the files where this script is run
    output_dir = "./json_outputs"  
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for angle in angles:
        # First version: material1 = "BlueGreen", material2 = "Blue"
        data_1 = generate_json(angle, "BlueGreen", "Blue")
        filename_1 = f"BinaryChoice_constantSize_{angle}deg_BlueGreenCylinder_BlueCylinder.json"
        
        # Second version: material1 = "Blue", material2 = "BlueGreen"
        data_2 = generate_json(angle, "Blue", "BlueGreen")
        filename_2 = f"BinaryChoice_constantSize_{angle}deg_BlueCylinder_BlueGreenCylinder.json"
        
        # Write out the two JSON files
        with open(os.path.join(output_dir, filename_1), 'w') as f:
            json.dump(data_1, f, indent=2)
        
        with open(os.path.join(output_dir, filename_2), 'w') as f:
            json.dump(data_2, f, indent=2)

    print(f"Done! JSON files have been generated in: {output_dir}")

if __name__ == "__main__":
    main()
