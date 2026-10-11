# Recording a walkthrough for reconstruction

The first evaluation should be one real room, filmed as a continuous prerecorded video. Camera-pose estimation needs recognizable detail in overlapping views. Dense reconstruction also needs several views of the same visible surfaces.

## Before recording

- Clean the camera lens and use the main rear camera.
- Use a normal wide or 1× lens; avoid digital zoom and switching lenses mid-clip.
- Turn on steady room lighting. Avoid flickering lamps, strong backlighting, and sudden exposure changes.
- If possible, clear people, pets, screens, and other things that move independently of the room.
- Record at a sensible high resolution (1080p is a good starting point). A very high frame rate is unnecessary; 30 FPS is fine.

## Walk slowly

- Walk at a steady pace and keep the phone stable. Pause briefly before turning toward another wall.
- Keep each new view substantially overlapped with the previous view. Small translations are more useful than fast pans.
- Move around furniture and other important objects so their sides and corners appear from several angles.
- Walk a loop and revisit a few earlier views. This gives feature matching a chance to reconnect the camera path.
- Include room corners, doors, windows, and textured objects. Do not spend long stretches facing a blank wall or ceiling.
- Hold the camera at roughly the same height, then make a second slow pass at a different height if tabletops or low furniture matter.
- Keep exposure and focus stable. Avoid motion blur by moving slowly rather than relying on aggressive digital stabilization.

A 30–90 second clip is a useful first target. It is better to capture a careful loop of one room than a long, fast tour of many spaces. Keep the original file unchanged so the metadata and capture settings remain available.

## After recording

Copy the original video into `inputs/room.mp4`. Start with the keyframe-only pass and inspect `outputs/room/diagnostics/keyframes-contact-sheet.jpg` and `keyframes.csv`. Look for selected views that steadily move around the room, remain sharp, and overlap. If the selected images skip a wall or object, adjust the capture or try a denser selection before evaluating SfM.
