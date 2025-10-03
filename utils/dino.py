# utils/dino.py
#the student_model and teacher_model above should be wrappers that produce projection outputs suitable for DINO (normalized vectors). I expect you will pass ProjectionHead instances (student and teacher) or small models that map backbone features to projections. The code demonstrates momentum update and centering; you'll need to use teacher_forward outputs, subtract center, apply temperature sharpening per DINO recipe when computing loss.
import torch
import torch.nn as nn

class DINOTeacherStudent(nn.Module):
    def __init__(self, student_model: nn.Module, teacher_model: nn.Module, momentum=0.995, student_temp=0.1, teacher_temp=0.04):
        super().__init__()
        self.student = student_model
        self.teacher = teacher_model
        self.momentum = momentum
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp

        # initialize teacher params = student params
        for p_s, p_t in zip(self.student.parameters(), self.teacher.parameters()):
            p_t.data.copy_(p_s.data)
            p_t.requires_grad = False  # teacher not trained by gradient

        self.register_buffer("center", torch.zeros(1, self.student.model_proj_dim if hasattr(self.student, 'model_proj_dim') else 256))

    @torch.no_grad()
    def update_teacher(self):
        # EMA update
        m = self.momentum
        for p_s, p_t in zip(self.student.parameters(), self.teacher.parameters()):
            p_t.data = p_t.data * m + p_s.data * (1. - m)

    @torch.no_grad()
    def update_center(self, teacher_outputs, momentum=0.9):
        # teacher_outputs: [B, D]
        batch_center = torch.mean(teacher_outputs, dim=0, keepdim=True)
        self.center = self.center * momentum + batch_center * (1. - momentum)

    def forward_student(self, *args, **kwargs):
        return self.student(*args, **kwargs)

    @torch.no_grad()
    def forward_teacher(self, *args, **kwargs):
        return self.teacher(*args, **kwargs)

