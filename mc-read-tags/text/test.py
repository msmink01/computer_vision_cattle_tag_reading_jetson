from readTag import Options, create_digit_read_envir, webcam_img

opt = Options("best_accuracy.pth")

converter, model, AlignCollate_demo = create_digit_read_envir(opt)

