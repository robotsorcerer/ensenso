"""
Boxes are thus defined in (x,y) coordinate pairs

([Top-Left, Top-Right], [Bottom-Left, Bottom-Right])

for all three of faces, left eyes and right eyes
"""

face_boxes = [
	Detection("face_0_image.jpg", [501, 502, 882, 507], [502, 987, 882, 982]),
	Detection("face_1_image.jpg", [504, 501, 855, 500], [502, 998, 847, 997]),
    Detection("face_2_image.jpg", [524, 532, 822, 531], [523, 995, 826, 995]),
	Detection("face_3_image.jpg", [519, 506, 848, 502], [522, 974, 848, 978]),
	Detection("face_4_image.jpg", [534, 526, 830, 532], [530, 958, 838, 962]),
	Detection("face_5_image.jpg", [522, 528, 848, 526], [518, 974, 850, 972]),
	Detection("face_6_image.jpg", [477, 532, 802, 526], [477, 963, 802, 968]),
	Detection("face_7_image.jpg", [540, 518, 836, 514], [537, 980, 837, 976]),
	Detection("face_8_image.jpg", [446, 502, 776, 504], [448, 957, 778, 957]),
	Detection("face_9_image.jpg", [522, 518, 830, 518], [522, 764, 836, 962]),
	Detection("face_10_image.jpg", [448, 520, 783, 519], [453, 951, 783, 954]),
	Detection("face_11_image.jpg", [528, 519, 783, 519], [453, 951, 783, 954]),
	Detection("face_12_image.jpg", [393, 508, 732, 510], [388, 963, 730, 958]),
	]

left_eye_boxes
= [
	Detection("face_0_image.jpg", [574, 720, 651, 726], [570, 777, 651, 772]),
	Detection("face_1_image.jpg", [580, 722, 660, 722], [580, 768, 662, 781]),
    Detection("face_2_image.jpg", [579, 772, 646, 734], [588, 770, 652, 777]),
	Detection("face_3_image.jpg", [576, 729, 651, 723], [572, 783, 648, 780]),
	Detection("face_4_image.jpg", [576, 723, 648, 726], [584, 772, 642, 770]),
	Detection("face_5_image.jpg", [584, 729, 654, 730], [586, 766, 651, 770]),
	Detection("face_6_image.jpg", [508, 728, 564, 724], [506, 762, 566, 758]),
	Detection("face_7_image.jpg", [580, 734, 652, 738], [582, 776, 652, 766]),
	Detection("face_8_image.jpg", [476, 722, 532, 722], [476, 760, 537, 758]),
	Detection("face_9_image.jpg", [588, 732, 648, 730], [588, 770, 648, 765]),
	Detection("face_10_image.jpg", [471, 717, 537, 714], [471, 759, 532, 754]),
	Detection("face_11_image.jpg", [448, 520, 783, 519], [453, 951, 783, 954]),
	Detection("face_11_image.jpg", [448, 520, 783, 519], [453, 951, 783, 954]),
	Detection("face_12_image.jpg", [477, 710, 566, 706], [478, 764, 576, 768]),
	]

right_eye_boxes
= [
	Detection("face_0_image.jpg", [694, 728, 777, 728], [692, 765, 782, 768]),
	Detection("face_1_image.jpg", [688, 726, 771, 726], [688, 778, 782, 770]),
    Detection("face_2_image.jpg", [690, 724, 754, 722], [690, 766, 759, 766]),
	Detection("face_3_image.jpg", [688, 720, 778, 714], [681, 770, 774, 771]),
	Detection("face_4_image.jpg", [692, 729, 777, 728], [694, 768, 777, 764]),
	Detection("face_5_image.jpg", [687, 728, 782, 730], [688, 722, 777, 762]),
	Detection("face_6_image.jpg", [603, 722, 682, 724], [600, 762, 678, 760]),
	Detection("face_7_image.jpg", [694, 735, 768, 732], [686, 764, 771, 762]),
	Detection("face_8_image.jpg", [574, 723, 651, 718], [573, 764, 652, 759]),
	Detection("face_9_image.jpg", [699, 724, 764, 729], [696, 762, 764, 764]),
	Detection("face_10_image.jpg", [566, 722, 654, 723], [572, 762, 657, 764]),
	Detection("face_12_image.jpg", [415, 705, 460, 708], [414, 744, 460, 746]),
	]
