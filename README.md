# Animal-Recognition-with-Convolutional-Neuron-Networks

## In deep learning, a Convolutional Neural Network (CNN) is a class of Artificial Neural Network (ANN) that's commonly implemented in visual imagery analyzing. This algorithm is actually quite simple to understand, and we can classify this 6 x 6 image of the integer 2 below as an example:

<img width="284" alt="Screen Shot 2022-06-29 at 11 22 36 PM" src="https://user-images.githubusercontent.com/102645083/176606907-5dbb88b1-8a4b-4e26-bdf0-59deab69ad02.png">

## The first step is to design a filter/feature, which can be a 3 x 3 image, and match it with every combination of connected 3 x 3 portions from the original image, and there are 16 of them. The purpose is to identify the same feature in every part of this image.

<img width="212" alt="Screen Shot 2022-06-29 at 11 30 35 PM" src="https://user-images.githubusercontent.com/102645083/176608202-bf3637d0-5cb3-4cf4-bf20-cc41245597a6.png">

<img width="616" alt="Screen Shot 2022-06-29 at 11 30 48 PM" src="https://user-images.githubusercontent.com/102645083/176608226-968c8527-f4ab-4053-9a06-e6b3d73f7c10.png">

## We can think of each 3 x 3 portion as a matrix, where colored pixels are 1's and white pixels are 0's. By applying the dot product between every portion and the filter/feaure, we can create a feature map.
<img width="272" alt="Screen Shot 2022-06-29 at 11 34 33 PM" src="https://user-images.githubusercontent.com/102645083/176608769-f147e5fc-3e5f-4fdc-b936-96c08a2781d3.png">

## Like the process that every Artifical Neuron goes through, we add a bias value to the feature map (b = -1 in this example), and input the result into an activation function (sigmoid in this example).
<img width="428" alt="Screen Shot 2022-06-29 at 11 37 39 PM" src="https://user-images.githubusercontent.com/102645083/176609307-94164a19-2060-4916-bea8-7b48d9b2e21b.png">
<img width="389" alt="Screen Shot 2022-06-29 at 11 37 49 PM" src="https://user-images.githubusercontent.com/102645083/176609332-cb44ef92-3339-4b13-a01e-393fd86d7078.png">

## The next step is optional, and it can be Average or Max Pooling, which are compression methods. Occasionly, it's unnecessary to output every single data from the current layer to the next. We can divide the newly modified feature map into portions (four in this example), and take the average or maximum value (max in this example) from each portion. We can easily recognize which part of the image matches the most with the filter from this new feature map.
<img width="515" alt="Screen Shot 2022-06-29 at 11 48 17 PM" src="https://user-images.githubusercontent.com/102645083/176611115-7ef12719-d713-4401-8f9c-daab93366e6e.png">
<img width="173" alt="Screen Shot 2022-06-29 at 11 48 33 PM" src="https://user-images.githubusercontent.com/102645083/176611141-d8679a3d-acd2-44fd-9f1c-f23ac5807c18.png">





