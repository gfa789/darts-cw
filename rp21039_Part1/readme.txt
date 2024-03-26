To run code viola jones detections run "python dart.py" 
    - this will do every image by default
    - if you want to specify do 'python dart.py -n "./Dartboard/dart0.py"' for example
    - when run it will output data to "./data/data_all_images.csv"
    - will also produce an image "./viola_detected.jpg"

To run code for hough transformation run "python hough.py"
    - this will do every image by default 
        - if doing every image, outputs will go to "./detected/insert_file_name.jpg"
        - will also produce 2D Hough representations in "./houghs/insert_file_name.jpg"
        - also will produce image with hough lines drawn on with plotted center in "./lines/insert_file_name.jpg"
        - also will produce threshold images at "./threshold/insert_file_name.jpg"
        - data for tpr and f1 score will be saved to "./data/data_hough_vj.csv"

To run code for hough transformation of specific image run 'python dart.py -n "./Dartboard/dart0.py"' for example
    - this will produce an output image at "./detected.jpg"
    - also produce hough space image at "./hough_space.jpg"
    - also produce thresholded image at "./threshold_image.jpg"
    - finally will produce image with hough lines drawn on with plotted center at "lines_drawn.jpg"