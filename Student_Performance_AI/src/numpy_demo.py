import numpy as np

student_data = np.array([
    [92, 18, 85, 88],
    [85, 14, 76, 80],
    [70, 8, 55, 62],
    [95, 20, 92, 94]
])

print("Original array:")
print(student_data)
print("Shape:", student_data.shape)

reshaped = student_data.reshape(2, 8)
print("\nReshaped:")
print(reshaped)

attendance = student_data[:, 0]
study_hours = student_data[:, 1]
previous_marks = student_data[:, 2]

print("\nAttendance:", attendance)
print("Study hours:", study_hours)
print("Previous marks:", previous_marks)

x = np.array([0.90, 0.80, 0.85, 0.88])
w = np.array([0.30, 0.20, 0.25, 0.25])
bias = 0.05
print("\nDot product / weighted sum:", np.dot(x, w) + bias)
