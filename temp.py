from cnnClassifier.pipeline.prediction import PredictionPipeline

# 👇 put path of any test image here
img_path = "test.jpg"

pred = PredictionPipeline(img_path)
result = pred.predict()

print(result)
