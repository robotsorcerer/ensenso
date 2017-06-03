#! /bin/bash

cd raw/face_neg/neg_2

for x in *.png; do
	#statements
	convert $x "$x.jpg";
	rm *.png
done


# now rename
for x in *.jpg; do
	#statements
	mv $x "${x/png_data/fake2_}"
done

# rename pcds
for x in *.pcd;  do
	#statements
	mv $x "${x/pcd_data/fake2_}"
done


#do neg_3
cd raw/face_neg/neg_3

tar -zxf *.gz

for x in *.png; do
	#statements
	convert $x "$x.jpg";
done

#remove unneeded files
rm *.png


# now rename
for x in *png.jpg; do
	#statements
	mv $x "${x/.png.jpg/.jpg}"
done

for x in *.jpg; do	
	mv $x "${x/png_data/fake3_}"
done

# rename pcds
for x in *.pcd;  do
	#statements
	mv $x "${x/pcd_data/fake3_}"
done


#do neg_4
cd raw/face_neg/neg_4

tar -zxf *.gz

for x in *.png; do
	#statements
	convert $x "$x.jpg";
done

#remove unneeded files
rm *.png


# now rename
for x in *png.jpg; do
	#statements
	mv $x "${x/.png.jpg/.jpg}"
done

for x in *.jpg; do	
	mv $x "${x/png_data/fake4_}"
done

# rename pcds
for x in *.pcd;  do
	#statements
	mv $x "${x/pcd_data/fake4_}"
done