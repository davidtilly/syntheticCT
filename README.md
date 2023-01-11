# syntheticCT
Create synthetic CT images from CBCT

# DONE
- Improve exisiting deformable image registrations
    - Crop registration volume vs external in the structureset
    - Determine bspline dimensions from dimensions of external -> end up at resolution ~ 2 cm in the bspline grid? 

- Test elastix deformable image registrations
    - similar as above
    - rigidity with elastix
    - contour points to constrain bone registration

- Post processing
    - Use external and 2 cm band as mask to copy from original CT (prevent qui)

- Fix bug with rescle / intercept

# TODO
- Test elastix deformable image registrations
    - bone rigidity with elastix (use bones as mask for rigidity penalty)

