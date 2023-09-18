# from django import forms
#
#
# from django.core.exceptions import ValidationError
#
# # def validate_image(file):
# #     if file.content_type != 'csv':
# #         raise ValidationError("只允许上传csv格式的文件。")
#
# def validate_file_name(file):
#     validate_files = ["upload\\train_features(MINE).csv", "upload\\train_label（MINE）.csv",
#                       "upload\\test_features（MINE）.csv",
#                       "upload\\test_label(MINE).csv"]
#     if file.name not in validate_files:
#         raise ValidationError("上传的文件名错误，请重新上传！")
#
# class UploadFileForm(forms.Form):
#     file = forms.FileField(validators=[ validate_file_name])
#
