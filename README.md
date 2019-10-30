# eval4txt_pycocotools

Eval detection results in TXT format using pycocotools

Tutorial
---------------
* eval_det_txt_pycoco.py:

  eg: python eval_det_txt_pycoco.py anno_file txt_file category


  anno_file:  Absolute path of the annotation file(json format)
  
  txt_file:   Absolute path of the txt file
  
  category:   Category of detection results, default category(category_id):
              person(1) head(2) face(3).

             Change the main function corresponding to the GT if necessary