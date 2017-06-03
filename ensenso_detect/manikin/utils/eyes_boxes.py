"""
Boxes are thus defined in (x,y) coordinate pairs
for both left eye and right eye

Top-Left, Top-Right

Bottom-Left, Bottom-Right
"""


eyes_boxes
= [
	Detection("face_0_image.jpg", [501, 502, 882, 507], [502, 987, 882, 982]),
	Detection("face_1_image.jpg", [504, 501, 855, 500], [502, 998, 847, 997]),
    Detection("face_2_image.jpg", [524, 532, 822, 531], [523, 995, 826, 995]),
	Detection("face_3_image.jpg", [519, 506, 848, 502], [522, 974, 848, 978]),
	Detection("face_4_image.jpg", [501, 502, 882, 507], [502, 987, 882, 982])
	]