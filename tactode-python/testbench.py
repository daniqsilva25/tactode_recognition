import sys
import os
import time
import numpy as np
import cv2 as cv
import tensorflow as tf
import tactode_pipeline as pipeline
import classification
import params


# FUNCTIONS
def setup_save_dir(file_path="", folder_path=""):
    if os.path.exists(folder_path):
        for e in os.listdir(folder_path):
            if os.path.isfile(os.path.join(folder_path, e)):
                os.unlink(os.path.join(folder_path, e))
            else:
                os.rmdir(os.path.join(folder_path, e))
    else:
        os.mkdir(folder_path)
    if len(file_path) > 0:
        fd = open(file_path, "w")
        fd.close()


def init_confusions(file_path="", tactode_pieces_file=""):
    confusions = []
    aux_conf = []
    fd = open(tactode_pieces_file, "r")
    for ln in fd:
        if len(ln) > 2:
            pieces = ln.split("\n")[0].split(":")[1].split(",")
            for p in pieces:
                confusions.append(params.Confusions(p))
                aux_conf.append(p)
    for cf in confusions:
        for aux_cf in aux_conf:
            cf.conf_matrix.append(params.ConfMatrix(aux_cf))
    fd.close()
    fd = open(file_path, "w")
    fd.close()
    return confusions


def update_confusions(confusions_lst=[], expected_p="", obtained_p=""):
    for p_conf in confusions_lst:
        if p_conf.id == obtained_p:
            for cm in p_conf.conf_matrix:
                if cm.id == expected_p:
                    cm.count += 1
        if expected_p == obtained_p:
            if p_conf.id == obtained_p:
                p_conf.tp += 1
        else:
            if p_conf.id == obtained_p:
                p_conf.fp += 1
            elif p_conf.id == expected_p:
                p_conf.fn += 1


def get_expected_pieces(filename="", folder_path=""):
    pieces_list = []
    num_pieces = 0
    code_txt_file = "%s.txt" % os.path.join(folder_path, filename)
    if os.path.exists(code_txt_file):
        fd = open(code_txt_file, "r")
        for ln in fd:
            if len(ln) > 2:
                ln_content = ln.split("\n")[0]
                ln_pieces = ln_content.split(",")
                num_pieces += len(ln_pieces)
                pieces_list.append(ln_pieces)
        fd.close()
    else:
        print("ERROR: The file '%s' does not exist" % code_txt_file)
    return params.PiecesList(pieces_list, num_pieces, False)


def get_pieces_evaluation(confusions_lst=[], obtained=params.PiecesList(), expected=params.PiecesList()):
    num_errors = 0
    errors_list = []
    if obtained.is_rotated:
        num_errors = -2
    elif obtained.num_pieces == expected.num_pieces:
        for i in range(len(obtained.list)):
            for j in range(len(obtained.list[i])):
                obtained_piece = obtained.list[i][j]
                expected_piece = expected.list[i][j]
                if obtained_piece != expected_piece:
                    num_errors += 1
                    errors_list.append(params.PieceError(i, j, expected_piece, obtained_piece))
                update_confusions(confusions_lst, expected_piece, obtained_piece)
    else:
        num_errors = -1
    return params.ErrorsList(errors_list, num_errors)


def write_evaluation(eval_file="", target_img="", evaluation=params.ErrorsList()):
    file_content = "%s\n" % target_img
    if evaluation.num_errors == 0:
        file_content += "  [YES] -> Finished with success!\n"
    elif evaluation.num_errors == -1:
        file_content += "  [NO_1] -> Number of pieces does not match!\n"
    elif evaluation.num_errors == -2:
        file_content += "  [NO_2] -> The code is rotated!\n"
    else:
        file_content += "  [NO_3] -> Some pieces are mismatched!\n"
        for item in evaluation.list:
            file_content += "    (%d,%d): Expected = '%s' ... Obtained = '%s'\n" \
                            % (item.i, item.j, item.expected_piece, item.obtained_piece)
    ef = open(eval_file, "a")
    ef.write("%s\n\n" % file_content)
    ef.close()


def write_confusions_file(confusions_file="", confusions=[]):
    file_content = ""
    for c in confusions:
        str_conf_matrix = ""
        for cm in c.conf_matrix:
            str_conf_matrix += "'%s':'%d'" % (cm.id, cm.count)
            if c.conf_matrix.index(cm) < len(c.conf_matrix) - 1:
                str_conf_matrix += " , "
        file_content += "'%s' -> { tp:%d , fp:%d , fn:%d , { %s } }\n\n" % (c.id, c.tp, c.fp, c.fn, str_conf_matrix)
    cf = open(confusions_file, "a")
    cf.write(file_content)
    cf.close()


def write_confusion_matrix(conf_matrix_file="", confusions=[]):
    file_content = ""
    for c in confusions:
        for cm in c.conf_matrix:
            file_content += "%d" % cm.count
            if c.conf_matrix.index(cm) != len(c.conf_matrix) - 1:
                file_content += ","
        file_content += "\n"
    cmf = open(conf_matrix_file, "a")
    cmf.write(file_content)
    cmf.close()


def write_latex_table_of_confusions(latex_table_file="", confusions=[]):
    file_content = ""
    for c in confusions:
        tile_name = c.id
        if tile_name.find("_") != -1:
            parts = tile_name.split("_")
            tile_name = "%s %s" % (parts[0], parts[1])
        file_content += "%s " % tile_name
        tp = c.tp
        fp = c.fp
        fn = c.fn
        file_content += "& %d & %d & %d " % (tp, fp, fn)
        precision = "-" if tp == 0 and fp == 0 else np.floor(tp * 1000 / (tp + fp)) / 1000
        recall = "-" if tp == 0 and fn == 0 else np.floor(tp * 1000 / (tp + fn)) / 1000
        f1_score = "-" if precision == 0 and recall == 0 or precision == "-" or recall == "-" \
            else np.floor(2 * precision * recall * 1000 / (precision + recall)) / 1000
        if precision != "-":
            file_content += "& %.3f " % precision
        else:
            file_content += "& %s " % precision
        if recall != "-":
            file_content += "& %.3f " % recall
        else:
            file_content += "& %s " % recall
        if f1_score != "-":
            file_content += "& %.3f " % f1_score
        else:
            file_content += "& %s " % f1_score
        file_content += "\\\\ \\hline \n"
    ltf = open(latex_table_file, "a")
    ltf.write(file_content)
    ltf.close()


# MAIN PROGRAM
print("\ntest started: %s\n" % params.get_current_time())

tm_fddm = params.TemplateMatchFeatDetDesc()
hog_svm_detection = params.HogSvmDetector()
hog_detector = cv.HOGDescriptor(hog_svm_detection.hog_file)
svm_detector = cv.ml.SVM_load(hog_svm_detection.svm_file)

test_quantity = ""
if len(sys.argv) > 1:
    test_quantity = sys.argv[1]

if test_quantity == "all":
    classification_dataset = ''
    main_results_folder = ''
    if len(sys.argv) > 2:
        if sys.argv[2] == 'small':
            classification_dataset = params.DATASET['test_small']
            main_results_folder = params.RESULTS['folder'] + 'small'
        elif sys.argv[2] == 'big':
            classification_dataset = params.DATASET['test_big']
            main_results_folder = params.RESULTS['folder'] + 'big'
        else:
            print("\nERROR: You must specify a VALID test dataset (small/big) to use!\n")
            exit(-1)
    else:
        print("\nERROR: You must specify the test dataset (small/big) to use!\n")
        exit(-1)

    results_file = main_results_folder + '/' + params.RESULTS['file']
    fres = open(results_file, "w")
    res_file_content = ""
    res_file_content += "Caption:\n"
    res_file_content += ".NoE3  = Number of misclassified pieces\n"
    res_file_content += ".NoI   = Number of Images tested\n"
    res_file_content += ".NoP   = Number of Pieces tested\n"
    res_file_content += ".NoUP  = Number of Unclassified Pieces\n"
    res_file_content += ".Acc   = Accuracy\n"
    res_file_content += ".AET   = Average Execution Time\n"
    res_file_content += ".SDET  = Standard Deviation Execution Time\n"
    res_file_content += ".minET = Minimum Execution Time\n"
    res_file_content += ".maxET = Maximum Execution Time\n\n"
    fres.write(res_file_content)
    fres.close()
    fc = open(params.TESTS['config'], "r")
    methods_list = []
    for lf in fc:
        if len(lf) > 0:
            methods_list.append(lf.split("\n")[0])
    fc.close()
    for m in methods_list:
        res_file_content = ""
        classif_method = m
        if classif_method == "HOG" \
                or classif_method == "VGG16" or classif_method == "VGG19"\
                or classif_method == "MobileNetV2" or classif_method == "ResNet152"\
                or classif_method == "YOLO" or classif_method == "SSD" \
                or classif_method == "ORB" or classif_method == "SURF" or \
                classif_method == "BRISK" or classif_method == "SIFT" or \
                classif_method == "TM_CCOEFF" or classif_method == "TM_SQDIFF":
            print("\n\n <------- %s | %s ------->\n" % (classif_method, params.get_current_time()))

            if classif_method == "HOG":
                hog_svm_classification = params.HogSvmClassifier()
                hog_svm_classification.load_classes()
                # Load HOG and SVM configurations
                hog_classifier = cv.HOGDescriptor(hog_svm_classification.hog_file)
                svm_classifier = cv.ml.SVM_load(hog_svm_classification.svm_file)
            elif classif_method == "YOLO":
                yolo = params.YOLO()
                yolo.load_classes()
                # Load the YOLO network
                yolo_net = cv.dnn.readNetFromDarknet(yolo.cfg_file, yolo.weights_file)
            elif classif_method == "SSD":
                ssd = params.SSD()
                ssd.load_classes()
                # Load graph
                with tf.python.gfile.GFile(ssd.model_file, "rb") as f:
                    graph_def = tf.compat.v1.GraphDef()
                    graph_def.ParseFromString(f.read())
                sess = tf.compat.v1.Session()
                sess.graph.as_default()
                tf.import_graph_def(graph_def, name="")
            elif classif_method == "VGG16":
                vgg16 = params.VGG16()
                vgg16.load_classes()
                # Load VGG16 model
                model = tf.keras.models.load_model(vgg16.model_folder)
            elif classif_method == "VGG19":
                vgg19 = params.VGG19()
                vgg19.load_classes()
                # Load VGG19 model
                model = tf.keras.models.load_model(vgg19.model_folder)
            elif classif_method == "MobileNetV2":
                mobilenetV2 = params.MobileNetV2()
                mobilenetV2.load_classes()
                # Load MobileNetV2 model
                model = tf.keras.models.load_model(mobilenetV2.model_folder)
            elif classif_method == "ResNet152":
                resnet152 = params.ResNet152()
                resnet152.load_classes()
                # Load ResNet152 model
                model = tf.keras.models.load_model(resnet152.model_folder)
            elif classif_method == "ORB" or classif_method == "BRISK" or \
                    classif_method == "SIFT" or classif_method == "SURF":
                bf = cv.BFMatcher_create(normType=cv.NORM_HAMMING, crossCheck=True)
                if classif_method == "ORB":
                    fdd = cv.ORB_create(nfeatures=600)
                elif classif_method == "BRISK":
                    fdd = cv.BRISK_create(thresh=10)
                else:
                    bf = cv.BFMatcher_create(normType=cv.NORM_L1, crossCheck=True)
                    if classif_method == "SIFT":
                        fdd = cv.SIFT_create(nfeatures=600)
                    elif classif_method == "SURF":
                        fdd = cv.xfeatures2d.SURF_create(hessianThreshold=50)

            method_result_folder = os.path.join(main_results_folder, classif_method)
            if not os.path.exists(method_result_folder):
                os.mkdir(method_result_folder)

            confusions_file = os.path.join(method_result_folder, "confusions.txt")
            confusions = init_confusions(confusions_file, params.TESTS['tiles'])

            res_file_content += "<--- %s --->\n" % classif_method
            fres = open(results_file, "a")
            fres.write(res_file_content)
            fres.close()

            exec_times_list = []
            sum_exec_times = 0
            number_of_pics = 0
            number_of_pieces = 0
            global_number_of_pieces = 0
            num_total_errors = 0
            global_num_total_errors = 0
            num_codes_w_errors = 0
            global_num_codes_w_errors = 0
            num_errors_type1 = 0
            global_num_errors_type1 = 0
            num_errors_type2 = 0
            global_num_errors_type2 = 0
            num_errors_type3 = 0
            global_num_errors_type3 = 0
            num_unclassified_pieces = 0
            global_num_unclassified_pieces = 0

            for resolution in sorted(os.listdir(classification_dataset)):
                fres = open(results_file, "a")
                res_file_content = ""
                res_file_content += "  * %s -> " % resolution
                number_of_pieces = 0
                num_total_errors = 0
                num_codes_w_errors = 0
                num_errors_type1 = 0
                num_errors_type2 = 0
                num_errors_type3 = 0
                num_unclassified_pieces = 0
                print("    * testing %s ..." % resolution)

                resolution_test_folder = os.path.join(classification_dataset, resolution)
                resolution_result_folder = os.path.join(method_result_folder, resolution)
                if not os.path.exists(resolution_result_folder):
                    os.mkdir(resolution_result_folder)

                for folder in sorted(os.listdir(resolution_test_folder)):
                    codes_result_folder = os.path.join(resolution_result_folder, folder)
                    eval_codes_file = os.path.join(codes_result_folder, "aval-%s.txt" % folder)
                    setup_save_dir(eval_codes_file, codes_result_folder)

                    codes_test_folder = os.path.join(resolution_test_folder, folder)
                    expected_pieces = get_expected_pieces(folder, codes_test_folder)
                    if expected_pieces.num_pieces > 0:
                        for img in sorted(os.listdir(codes_test_folder)):
                            if img.endswith(".jpg"):

                                img_path = os.path.join(codes_test_folder, img)
                                start_time = time.time()
                                pipeline_res = pipeline.run_main(img_path, hog=hog_detector, svm=svm_detector,
                                                       config=hog_svm_detection)
                                if classif_method == "HOG":
                                    obtained_pieces = classification.run_hog_svm(
                                        svm=svm_classifier, hog=hog_classifier, config=hog_svm_classification, res=pipeline_res)
                                elif classif_method == "VGG16":
                                    obtained_pieces = classification.run_vgg16(model=model, config=vgg16, res=pipeline_res,
                                                                               tf=tf)
                                elif classif_method == "VGG19":
                                    obtained_pieces = classification.run_vgg19(model=model, config=vgg19, res=pipeline_res,
                                                                               tf=tf)
                                elif classif_method == "MobileNetV2":
                                    obtained_pieces = classification.run_mobilenet_v2(model=model, config=mobilenetV2,
                                                                                      res=pipeline_res, tf=tf)
                                elif classif_method == "ResNet152":
                                    obtained_pieces = classification.run_resnet152(model=model, config=resnet152,
                                                                                      res=pipeline_res, tf=tf)
                                elif classif_method == "YOLO":
                                    obtained_pieces = classification.run_yolo(net=yolo_net, config=yolo, res=pipeline_res)
                                elif classif_method == "SSD":
                                    obtained_pieces = classification.run_ssd(sess=sess, config=ssd, res=pipeline_res)
                                elif classif_method == "TM_CCOEFF" or classif_method == "TM_SQDIFF":
                                    obtained_pieces = classification.run_template_matching(
                                        tm_type=classif_method, config=tm_fddm, res=pipeline_res)
                                elif classif_method == "ORB" or classif_method == "BRISK" or classif_method == "SIFT" or classif_method == "SURF":
                                    obtained_pieces = classification.run_features_det_desc_match(
                                        fdd=fdd, bf=bf, config=tm_fddm, res=pipeline_res)
                                end_time = time.time()
                                exec_time = end_time - start_time
                                exec_times_list.append(exec_time)
                                sum_exec_times += exec_time
                                number_of_pics += 1
                                number_of_pieces += expected_pieces.num_pieces
                                evaluation = get_pieces_evaluation(confusions, obtained_pieces, expected_pieces)
                                if evaluation.num_errors != 0:
                                    num_codes_w_errors += 1
                                    if evaluation.num_errors > 0:
                                        num_errors_type3 += evaluation.num_errors
                                    elif evaluation.num_errors == -1:
                                        num_errors_type1 += 1
                                    elif evaluation.num_errors == -2:
                                        num_errors_type2 += 1
                                num_total_errors = num_errors_type1 + num_errors_type2 + num_errors_type3
                                for row in obtained_pieces.list:
                                    num_unclassified_pieces += row.count("--")
                                write_evaluation(eval_codes_file, img, evaluation)
                    else:
                        print("ERROR: Missing content on file '%s.txt'!" % folder)
                print("      > Number of codes with errors:", num_codes_w_errors)
                print("      > Number of errors of type 1:", num_errors_type1)
                print("      > Number of errors of type 2:", num_errors_type2)
                print("      > Number of errors of type 3:", num_errors_type3)
                print("      > Total amount of errors:", num_total_errors)
                print("      > Number of pieces tested:", number_of_pieces)
                print("      > Number of unclassified pieces:", num_unclassified_pieces)
                acc = (number_of_pieces - num_errors_type3) * 100 / number_of_pieces
                print("      > Accuracy (%%): %.2f\n" % acc)
                res_file_content += " NoE3: %d | NoUP: %d | NoP: %d | Acc: %.2f %%\n" \
                                    % (num_errors_type3, num_unclassified_pieces, number_of_pieces, acc)
                fres.write(res_file_content)
                fres.close()
                global_num_codes_w_errors += num_codes_w_errors
                global_num_errors_type1 += num_errors_type1
                global_num_errors_type2 += num_errors_type2
                global_num_errors_type3 += num_errors_type3
                global_num_total_errors += num_total_errors
                global_number_of_pieces += number_of_pieces
                global_num_unclassified_pieces += num_unclassified_pieces
            write_confusions_file(confusions_file, confusions)
            latex_table_file = os.path.join(method_result_folder, "latex-%s.txt" % classif_method)
            ltf = open(latex_table_file, "w")
            ltf.close()
            write_latex_table_of_confusions(latex_table_file, confusions)
            conf_matrix_file = os.path.join(method_result_folder, "conf_matrix-%s.csv" % classif_method)
            cmf = open(conf_matrix_file, "w")
            cmf.close()
            write_confusion_matrix(conf_matrix_file, confusions)
            print("  +++++ Global -> %s +++++" % classif_method)
            print("   Number of codes with errors:", global_num_codes_w_errors)
            print("   Number of errors of type 1:", global_num_errors_type1)
            print("   Number of errors of type 2:", global_num_errors_type2)
            print("   Number of errors of type 3:", global_num_errors_type3)
            print("   Total amount of errors:", global_num_total_errors)
            print("\n   Number of pictures tested:", number_of_pics)
            print("   Number of pieces tested:", global_number_of_pieces)
            print("   Number of unclassified pieces:", global_num_unclassified_pieces)
            global_accuracy = (global_number_of_pieces - global_num_errors_type3) * 100 / global_number_of_pieces
            avg_exec_time = sum_exec_times / number_of_pics
            sum_mean = 0
            for curr_et in exec_times_list:
                sum_mean += pow(curr_et - avg_exec_time, 2)
            std_deviation_exec_time = pow(sum_mean / number_of_pics, 0.5)
            max_exec_time = max(exec_times_list.__iter__())
            min_exec_time = min(exec_times_list.__iter__())
            print("   Accuracy (%%): %.2f" % global_accuracy)
            print("   Average execution time (seconds): %.3f" % avg_exec_time)
            print("   Standard deviation execution time (seconds): %.3f" % std_deviation_exec_time)
            print("   Minimum execution time (seconds): %.3f" % min_exec_time)
            print("   Maximum execution time (seconds): %.3f" % max_exec_time)
            print("\n !------- %s | %s -------!" % (classif_method, params.get_current_time()))
            res_file_content = ""
            res_file_content += "  GLOBAL %s\n" % classif_method
            res_file_content += "   - NoI: %d\n" % number_of_pics
            res_file_content += "   - NoP: %d\n" % global_number_of_pieces
            res_file_content += "   - NoUP: %d\n" % global_num_unclassified_pieces
            res_file_content += "   - NoE3: %d\n" % global_num_errors_type3
            res_file_content += "   - Acc: %.2f %%\n" % global_accuracy
            res_file_content += "   - AET: %.3f seconds\n" % avg_exec_time
            res_file_content += "   - SDET: %.3f seconds\n" % std_deviation_exec_time
            res_file_content += "   - minET: %.3f seconds\n" % min_exec_time
            res_file_content += "   - maxET: %.3f seconds\n\n" % max_exec_time
            fres = open(results_file, "a")
            fres.write(res_file_content)
            fres.close()
        else:
            print("\nERROR: Unrecognized classification method! -> %s" % classif_method)
elif test_quantity == "one":
    if len(sys.argv) != 4:
        print("\nERROR: You must specify the classification method "
              "(HOG/YOLO/SSD/VGG16/VGG19/MobileNetV2/ResNet152/ORB/BRISK/SIFT/SURF/TM_CCOEFF/TM_SQDIFF) "
              "and the image path!\n")
        exit(-1)
    else:
        classif_method = sys.argv[2]
        test_img_path = sys.argv[3]
        pipeline_res = pipeline.run_main(test_img_path, hog_detector, svm_detector, hog_svm_detection)
        is_valid = False
        if classif_method == 'HOG':
            is_valid = True
            hog_svm_classification = params.HogSvmClassifier()
            hog_svm_classification.load_classes()
            hog_classifier = cv.HOGDescriptor(hog_svm_classification.hog_file)
            svm_classifier = cv.ml.SVM_load(hog_svm_classification.svm_file)
            obtained_pieces = classification.run_hog_svm(
                svm=svm_classifier,
                hog=hog_classifier,
                config=hog_svm_classification,
                res=pipeline_res
            )
        elif classif_method == 'YOLO':
            is_valid = True
            yolo = params.YOLO()
            yolo.load_classes()
            yolo_net = cv.dnn.readNetFromDarknet(yolo.cfg_file, yolo.weights_file)
            obtained_pieces = classification.run_yolo(
                net=yolo_net,
                config=yolo,
                res=pipeline_res
            )
        elif classif_method == 'SSD':
            is_valid = True
            ssd = params.SSD()
            ssd.load_classes()
            with tf.python.gfile.GFile(ssd.model_file, "rb") as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
            sess = tf.compat.v1.Session()
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name="")
            obtained_pieces = classification.run_ssd(
                sess=sess,
                config=ssd,
                res=pipeline_res
            )
        elif classif_method == 'VGG16':
            is_valid = True
            vgg16 = params.VGG16()
            vgg16.load_classes()
            model = tf.keras.models.load_model(vgg16.model_folder)
            obtained_pieces = classification.run_vgg16(
                model=model,
                config=vgg16,
                res=pipeline_res,
                tf=tf
            )
        elif classif_method == 'VGG19':
            is_valid = True
            vgg19 = params.VGG19()
            vgg19.load_classes()
            model = tf.keras.models.load_model(vgg19.model_folder)
            obtained_pieces = classification.run_vgg19(
                model=model,
                config=vgg19,
                res=pipeline_res,
                tf=tf
            )
        elif classif_method == 'MobileNetV2':
            is_valid = True
            mnetv2 = params.MobileNetV2()
            mnetv2.load_classes()
            model = tf.keras.models.load_model(mnetv2.model_folder)
            obtained_pieces = classification.run_mobilenet_v2(
                model=model,
                config=mnetv2,
                res=pipeline_res,
                tf=tf
            )
        elif classif_method == 'ResNet152':
            is_valid = True
            resnet152 = params.ResNet152()
            resnet152.load_classes()
            model = tf.keras.models.load_model(resnet152.model_folder)
            obtained_pieces = classification.run_resnet152(
                model=model,
                config=resnet152,
                res=pipeline_res,
                tf=tf
            )
        elif classif_method == 'TM_CCOEFF' or classif_method == 'TM_SQDIFF':
            is_valid = True
            obtained_pieces = classification.run_template_matching(
                tm_type=classif_method,
                config=params.TemplateMatchFeatDetDesc(),
                res=pipeline_res
            )
        elif classif_method == 'ORB' or classif_method == 'BRISK' or \
                classif_method == 'SIFT' or classif_method == 'SURF':
            is_valid = True
            bf = cv.BFMatcher_create(cv.NORM_HAMMING, True)
            if classif_method == 'ORB':
                fdd = cv.ORB_create(nfeatures=600)
            elif classif_method == 'BRISK':
                fdd = cv.BRISK_create(thresh=10)
            else:
                bf = cv.BFMatcher_create(cv.NORM_L1, True)
                if classif_method == 'SIFT':
                    fdd = cv.SIFT_create(nfeatures=600)
                elif classif_method == 'SURF':
                    fdd = cv.xfeatures2d.SURF_create(hessianThreshold=50)
            obtained_pieces = classification.run_features_det_desc_match(
                fdd=fdd,
                bf=bf,
                config=params.TemplateMatchFeatDetDesc(),
                res=pipeline_res
            )
        if is_valid:
            classification.draw_result(pipeline_res[1], obtained_pieces.list, pipeline_res[0])
            classification.print_result(obtained_pieces)
        else:
            print("\nERROR: You must specify a valid classification method!\n")
            exit(-1)
else:
    print("ERROR: Please insert a valid option! ('one' or 'all')")
    exit(-1)

print("\ntest ended:   %s" % params.get_current_time())
