from deepface import DeepFace

metrics = ["cosine", "euclidean", "euclidean_l2"]

#face verification
result = DeepFace.verify(
  img1_path = "me1.jpg", 
  img2_path = "me2.jpg",model_name="Facenet", enforce_detection=False
)

print(result)