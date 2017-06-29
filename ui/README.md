Javascript visualization of Cuda kilobot simulator

This will display the contents of:
- `shapes.tsv`: Colored background rectangles
- `robots.tsv`: All kilobots

To run:
- Run `python -m SimpleHTTPServer` (starts a simple server to be able to load the `.tsv` files)
- Load in browser at `0.0.0.0:8000/ui/index.html` or wherever the server tells you to go

You can also use any other basic server, but Python is easy

The output displays at 1 px = 1 mm in simulation. The easiest way to see what you want is to zoom in your browser.

The goal is to live re-load the contents `robots.tsv`, which will be updated by the simulator. But it doesn't do that yet.

## TODO:

- Update when `robots.tsv` changes
- Some way to interact/pause/change speed? (probably needs to happen on simulator end because this just recieves)
- Maybe show time somewhere (below arena?)