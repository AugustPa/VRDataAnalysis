import json
import random

def generate_sequences(sequence_steps, intertrial_scene, repetitions):
    sequences = []
    for _ in range(repetitions):
        randomized_steps = sequence_steps.copy()
        random.shuffle(randomized_steps)
        for step in randomized_steps:
            sequences.append(intertrial_scene)
            sequences.append(step)
    return sequences

def main():
    # Define the sequence steps
    sequence_steps = [
        {
            "sceneName": "Choice_noBG",
            "duration": 45,
            "parameters": {
                "configFile": "BinaryChoice_constantSize_30deg_BlackCylinder_BlackCylinder.json"
            }
        },
        {
            "sceneName": "Choice_noBG",
            "duration": 45,
            "parameters": {
                "configFile": "BinaryChoice_constantSize_30deg_BlackCylinder_BlackCylinder.json"
            }
        },
        {'duration': 45,
        'parameters': {'configFile': 'BinaryChoice_constantSize_20deg_BlueGreenCylinder_BlueCylinder.json'},
        'sceneName': 'Choice_noBG'},
        {'duration': 45,
        'parameters': {'configFile': 'BinaryChoice_constantSize_20deg_BlueCylinder_BlueGreenCylinder.json'},
        'sceneName': 'Choice_noBG'},
        {'duration': 45,
        'parameters': {'configFile': 'BinaryChoice_constantSize_30deg_BlueGreenCylinder_BlueCylinder.json'},
        'sceneName': 'Choice_noBG'},
        {'duration': 45,
        'parameters': {'configFile': 'BinaryChoice_constantSize_30deg_BlueCylinder_BlueGreenCylinder.json'},
        'sceneName': 'Choice_noBG'},
        {'duration': 45,
        'parameters': {'configFile': 'BinaryChoice_constantSize_50deg_BlueGreenCylinder_BlueCylinder.json'},
        'sceneName': 'Choice_noBG'},
        {'duration': 45,
        'parameters': {'configFile': 'BinaryChoice_constantSize_50deg_BlueCylinder_BlueGreenCylinder.json'},
        'sceneName': 'Choice_noBG'},
        {'duration': 45,
        'parameters': {'configFile': 'BinaryChoice_constantSize_70deg_BlueGreenCylinder_BlueCylinder.json'},
        'sceneName': 'Choice_noBG'},
        {'duration': 45,
        'parameters': {'configFile': 'BinaryChoice_constantSize_70deg_BlueCylinder_BlueGreenCylinder.json'},
        'sceneName': 'Choice_noBG'},
        {'duration': 45,
        'parameters': {'configFile': 'BinaryChoice_constantSize_90deg_BlueGreenCylinder_BlueCylinder.json'},
        'sceneName': 'Choice_noBG'},
        {'duration': 45,
        'parameters': {'configFile': 'BinaryChoice_constantSize_90deg_BlueCylinder_BlueGreenCylinder.json'},
        'sceneName': 'Choice_noBG'},
        {'duration': 45,
        'parameters': {'configFile': 'BinaryChoice_constantSize_110deg_BlueGreenCylinder_BlueCylinder.json'},
        'sceneName': 'Choice_noBG'},
        {'duration': 45,
        'parameters': {'configFile': 'BinaryChoice_constantSize_110deg_BlueCylinder_BlueGreenCylinder.json'},
        'sceneName': 'Choice_noBG'},
        {'duration': 45,
        'parameters': {'configFile': 'BinaryChoice_constantSize_130deg_BlueGreenCylinder_BlueCylinder.json'},
        'sceneName': 'Choice_noBG'},
        {'duration': 45,
        'parameters': {'configFile': 'BinaryChoice_constantSize_130deg_BlueCylinder_BlueGreenCylinder.json'},
        'sceneName': 'Choice_noBG'},
        {'duration': 45,
        'parameters': {'configFile': 'BinaryChoice_constantSize_150deg_BlueGreenCylinder_BlueCylinder.json'},
        'sceneName': 'Choice_noBG'},
        {'duration': 45,
        'parameters': {'configFile': 'BinaryChoice_constantSize_150deg_BlueCylinder_BlueGreenCylinder.json'},
        'sceneName': 'Choice_noBG'},
        {'duration': 45,
        'parameters': {'configFile': 'BinaryChoice_constantSize_180deg_BlueGreenCylinder_BlueCylinder.json'},
        'sceneName': 'Choice_noBG'},
        {'duration': 45,
        'parameters': {'configFile': 'BinaryChoice_constantSize_180deg_BlueCylinder_BlueGreenCylinder.json'},
        'sceneName': 'Choice_noBG'}
    ]

    # Define the intertrial scene
    intertrial_scene = {
        "sceneName": "Forrest_choice",
        "duration": 15,
        "parameters": {
            "configFile": "Treecircle.json"
        }
    }

    # Get the number of repetitions from the user
    repetitions = int(input("Enter the number of repetitions: "))

    # Generate the sequences
    sequences = generate_sequences(sequence_steps, intertrial_scene, repetitions)

    # Create the final experiment configuration
    experiment_config = {
      "sequences": sequences
    }

    # Save the configuration to a JSON file
    with open("/home/flyvr01/Downloads/experiment_config_colorTest.json", "w") as json_file:
        json.dump(experiment_config, json_file, indent=4)

    print("Experiment configuration has been saved to experiment_config_colorTest.json .")

if __name__ == "__main__":
    main()
