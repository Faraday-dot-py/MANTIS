**First some housekeeping...**

- [x] Remove the need to create JSON files

- [x] Write a function to combine the output images into a single image before saving

- [x] Increase modularity
  - [x] File paths for all functions

- [ ] If ``createTemporalMapFromDir`` is passed an output dir, save the tmap to that dir and return it, otherwise just return it

- [x] Rename "testImage" and similar nomenclature relics

- [x] Better automation for data collection


**And after all that...**

- [x] Preliminary test with two image classes (r2l and l2r)
  - [x] Collect l2r and r2l data (20 samples of each)
  - [x] Generate tmaps for each
  - [x] Label images
  - [x] Train image classifier
  - [x] ✨Testing✨

If testing goes well, we can reach out to ASL foundations for data collection

Testing went well! Basic model with 2 classes (r2l and l2r) works
r2l: slow hand movement from right to left
l2r: slow hand movement from left to right

- [ ] Optimization
 - [ ] Run camera and processing in two different threads
 - [ ] Look into making the model smaller