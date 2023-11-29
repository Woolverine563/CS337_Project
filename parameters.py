from model import EncoderParams, ProjectorParams

class Parameter:
    def __init__(self,batch_size, target_batch_size, num_classes, input_length, name):
        self.batch_size = batch_size
        self.target_batch_size = target_batch_size
        self.num_classes = num_classes
        self.EncoderParams = EncoderParams(input_length)
        self.projectorParams = ProjectorParams(input_length)
        self.name = name