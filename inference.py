
import cv2
import time
from modelDSUNET import *
# Prepare input image
dummy_input = np.random.rand(1, 256, 256, 3).astype(np.float32)



image = cv2.imread(r"/home/manik/Documents/datasets/casia_dataset/val/image/COCO_DF_C110B00000_00002222.jpg")  # Load image
# image = tf.random.normal((1, 256, 256, 3))
image = cv2.resize(image, (256, 256))  # Resize to match input shape
image = image / 255.0  # Normalize to [0,1]
image = np.expand_dims(image, axis=0)  # Add batch dimension (1, 256, 256, 3)

_ = DSUNET_model.predict([dummy_input,dummy_input])

start_time = time.time()
output = DSUNET_model.predict([image, image])  # DS-UNet requires (RGB, Noise) input
print("Inference Time:", time.time() - start_time, "seconds")
output_mask = (output[0, :, :, 0] > 0.5).astype(np.uint8)  # Threshold at 0.5



cv2.imshow("Output Mask", output_mask * 255)
cv2.waitKey(0) 
cv2.destroyAllWindows()  

