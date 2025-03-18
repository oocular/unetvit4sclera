# SBVPI (Sclera Blood Vessels, Periocular and Iris) 

"SBVPI is a publicly available dataset designated primarily for research into sclera recognition, but it is also suitable for experiments with iris and periocular recognition techniques. It consists of 1858 high-resolution RGB images of eyes from 55 subjects. Images of the dataset were captured during a single recording session with a digital single-lens reflex camera (DSLR) (Canon EOS 60D) at the highest resolution and quality setting. Macro lenses were used to ensure sufficiently high-quality and visibility of fine details in the captured images. This procedure resulted in images with clearly visible scleral vasculature patterns, as shown in Figure 1." 

"Each sample in the database is labelled with an identity (one of 55), age (15–80), gender (male or female), eye (left or right), gaze direction (left, right, up, straight) and eye colour (brown, green, blue). Additionally, all 1858 images contain a manually-generated pixel-level ground truth markup of the sclera and periocular regions, as illustrated in Figure 3. A subset of 100–130 images (depending on the region) also contains a markup of the scleral vessels, the iris, the canthus, eyelashes, and the pupil. We used the GNU Image Manipulation Program (GIMP) to generate the ground-truth markups, which we included in SBVPI as separate images. Overall, the annotation process alone required roughly 500 man-hours of focused and accurate manual work. To conserve space, we later converted the markups into binary mask images, which (using lossless PNG compression) reduced the size of the entire dataset by a factor of approximately 6×. These masks are included in the final publicly available version of the dataset."

https://sclera.fri.uni-lj.si/datasets.html

![fig](Vasc.png)
Figure 1. Sample image from SBVPI with a zoomed in region that shows the sclera vasculature.

![fig](Annotations.png)
Figure 3: Each image has a set of corresponding per-pixel annotations (from left to right, top to bottom): RGB image, sclera, iris, vascular structures, pupil and periocular region.

## Obtaining the datasets
To obtain the SBVPI dataset, please download, fill out, and hand-sign this [form](https://docs.google.com/document/d/1HhR0T5qhzipRxUeDspZmqWiR5OGGT10yTnPDfnfQeBw/edit?tab=t.0), and send it to Matej at matej.vitek@fri.uni-lj.si.


