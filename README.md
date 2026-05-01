# Enhanced Low Cost Indoor Gardening System for Urban Food Sustainability

## Description
The Convolutional Neural Network component of the Enhanced Low Cost Indoor Gardening System for Urban Sustainability capstone project. Takes vegitation indexed images of individual plants as input and outputs values for the plant's hydration, nutrition, and lighting (range 0.0 - 1.0).

## Steps for Use
1. Name images in this manner; “5_digit_ID - 6_digit_date - vegetation_index"
2. Generate labels in this .csv row format; “5_digit_ID, hydration_value, nutrition_value, lighting_value”
3. Place the RGB, SIPI, and GNDVI images into the /data/Indices folder
4. Place the .csv label file into /data/Labels
  - Make sure in build_model.py the csv_labels_filename variable has the same name as the .csv file
5. Run image_transformer.py and check /data/Numpy to see if the matrices were created
6. Run main.py and choose the train option, assuming everything is correct the model will be trained
  - You can save the model for later with the save option (it will save into tests/models for now)
  - You can load the model for testing with the load option (loads from the same directory as the save option)
  - Test the loaded model with the test option (make sure to use a .npy file for testing!)

## License
MIT License Copyright (c) 2025 <leonarjd@miamioh.edu>

## Contact
Academic Advisor Email: <leonarjd@miamioh.edu>

Main Programmer Email: <meekergj@miamioh.edu>
