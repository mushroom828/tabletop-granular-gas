# Third-party notices

Unless a file states otherwise, the original project source code, including
the training, extraction, calibration, and analysis scripts, is licensed
under the GNU Affero General Public License version 3 only
(`AGPL-3.0-only`); see [LICENSE](LICENSE). Commercial use is permitted subject
to compliance with the applicable license terms. Any copyright in original
project source code remains with its respective authors. Third-party
components retain their own copyrights and licenses.

## SORT

- Bundled file: [`2_GetResults/sort.py`](2_GetResults/sort.py)
- Upstream project: <https://github.com/abewley/sort>
- Copyright (C) 2016-2020 Alex Bewley <alex@bewley.ai>
- License: GNU General Public License version 3 or any later version
  (`GPL-3.0-or-later`). A copy of version 3 is provided in
  [LICENSES/GPL-3.0.txt](LICENSES/GPL-3.0.txt).

The copyright and license notice in `sort.py` is retained. The project's
AGPL license does not replace the GPL license on that file. Combination of
GPLv3 and AGPLv3 code is addressed by section 13 of the respective licenses.
The bundled copy defers demo-only imports and random-seed initialization
to its `__main__` block, and loads visualization imports only for
`--display`. A project adaptation notice was added on 2026-11-08.

## Ultralytics YOLOv8 and model weights

- Upstream project: <https://github.com/ultralytics/ultralytics>
- License information: <https://www.ultralytics.com/license>
- The project uses the GNU AGPL version 3 open-source licensing option.
  Ultralytics code retains its upstream copyright and license notices.
- Bundled pretrained weights: `1_TrainYolov8/yolov8n.pt`.
- Project-trained detector: `2_GetResults/best.pt`.

Under Ultralytics' published licensing guidance, its pretrained and
fine-tuned models are subject to the applicable Ultralytics license terms.
The model weights are not relicensed as experimental data by this project.
The supplied `best.pt` records Ultralytics 8.4.210; `requirements.txt`
records the environment used for the release check, which can differ from
the training environment.

## Other dependencies

The external packages listed in `requirements.txt` retain their own
licenses. The project license
does not replace those licenses. If redistributing dependency packages or
their source, preserve the copyright and license notices supplied with
the exact versions being redistributed.

## Experimental materials

The software licenses described above apply to source code. They do not
by themselves grant a license for experimental data, images, annotations,
video recordings, figures, or manuscript text. Any separate license
expressly attached to those materials governs their reuse.
