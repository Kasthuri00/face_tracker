from insightface.app import FaceAnalysis

class FaceEmbedder:
    def __init__(self):
        self.model = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
        self.model.prepare(ctx_id=0)

    def get_embedding(self, frame, box):
        faces = self.model.get(frame)
        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox)
            if abs(x1 - box[0]) < 20 and abs(y1 - box[1]) < 20:
                return face.embedding
        return None
