
from defgrid.config import add_defgrid_maskhead_config
from detectron2.config import get_cfg
from predictor import VisualizationDemo
from detectron2.data.detection_utils import read_image

def get_image(demo, path)
    # use PIL, to be consistent with evaluation
    img = read_image(path, format="BGR")
    start_time = time.time()
    predictions, visualized_output = demo.run_on_image(img)
    logger.info(
        "{}: {} in {:.2f}s".format(
            path,
            "detected {} instances".format(
                len(predictions["instances"]))
            if "instances" in predictions
            else "finished",
            time.time() - start_time,
        )
    )

def visualization_demo(cfg, input, output, window_name):
    demo = VisualizationDemo(cfg)

    if input:
        if len(input) == 1:
            input = glob.glob(os.path.expanduser(input[0]))
            assert input, "The input path(s) was not found"
        for path in tqdm.tqdm(input, disable=not output):
            get_image(demo, path)

            if output:
                if os.path.isdir(output):
                    assert os.path.isdir(output), output
                    out_filename = os.path.join(
                        output, os.path.basename(path))
                else:
                    assert len(
                        input) == 1, "Please specify a directory with output"
                    out_filename = output
                visualized_output.save(out_filename)
            else:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow(
                    window_name, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
