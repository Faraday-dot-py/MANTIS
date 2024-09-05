**First some housekeeping...**

- [x] Remove the need to create JSON files

- [x] Write a function to combine the output images into a single image before saving

- [x] Increase modularity
  - [x] File paths for all functions

- [ ] If ``createTemporalMapFromDir`` is passed an output dir, save the tmap to that dir and return it, otherwise just return it

- [x] Rename "testImage" and similar nomenclature relics

- [x] Better automation for data collection


**And after all that...**

- [ ] Preliminary test with two image classes (r2l and l2r)
  - [ ] Collect l2r and r2l data (20 samples of each)
  - [ ] Generate tmaps for each
  - [ ] Label images
  - [ ] Train image classifier
  - [ ] ✨Testing✨

If testing goes well, we can reach out to ASL foundations for data collection
