# Command line interface

```sh
adhipk-ascii-art IMAGE_PATH -o result.png
adhipk-ascii-art IMAGE_PATH --columns 100 --font-size 12 -o result.png
adhipk-ascii-art IMAGE_PATH --no-edge -o tones.png
adhipk-ascii-art IMAGE_PATH --edge-threshold 50 -o edges.png
adhipk-ascii-art IMAGE_PATH --color --background '#101010' --foreground '#ffffff' -o color.png
adhipk-ascii-art IMAGE_PATH --text --columns 100 -o output.txt
```

Edges are enabled by default. `--edge-threshold` is a percentile from 0 to 100;
raising it selects fewer strong edges. `--font-path` accepts a TrueType/OpenType
font. Without a font path, Pillow supplies its bundled font. `--columns` fixes
character count while preserving image proportions through measured cell metrics.
`--foreground` and `--background` accept RGB hex strings. Input/output image
failures and invalid settings exit with code 2.

Run `adhipk-ascii-art --help` for the complete argument list.
