import numpy as np
import PIL.Image
import matplotlib
from matplotlib import pylab as P

from mpl_toolkits.axes_grid1 import make_axes_locatable

# package https://github.com/PAIR-code/saliency
import saliency

# Boilerplate methods.
def ShowImage(im, title='', ax=None):
    if ax is None:
        P.figure()
    P.axis('off')
    #im = ((im + 1) * 127.5).astype(np.uint8)
    im = (im * 127.5).astype(np.uint8)
    P.imshow(im)
    P.title(title)

def ShowGrayscaleImage(im, title='', ax=None):
    if ax is None:
        P.figure()
    P.axis('off')

    P.imshow(im, cmap=P.cm.gray, vmin=0, vmax=1)
    P.title(title)

def ShowHeatMap(im, title, ax=None):
    if ax is None:
        P.figure()
    P.axis('off')
    P.imshow(im, cmap='inferno')
    P.title(title)

def ShowDivergingImage(grad, title='', percentile=99, ax=None):
    if ax is None:
        fig, ax = P.subplots()
    else:
        fig = ax.figure

    P.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax.imshow(grad, cmap=P.cm.coolwarm, vmin=-1, vmax=1)
    fig.colorbar(im, cax=cax, orientation='vertical')
    P.title(title)

def LoadImage(file_path):
    im = PIL.Image.open(file_path)
    im = np.asarray(im)
    return im / 127.5 - 1.0


_ROWS, _COLS, _UPSCALE_FACTOR = 1, 2, 10

def plot_vanilla_gradient_mask(graph, sess, NN_model, img, img_class, img_class_name):
    pred, prob = sess.run([NN_model.pred, NN_model.prob], feed_dict={NN_model.model_input: [img]})
    gradient_saliency = saliency.GradientSaliency(graph, sess, NN_model.y, NN_model.model_input)

    # Get 3D mask
    vanilla_mask_3d = gradient_saliency.GetMask(img, feed_dict = {NN_model.class_selector: img_class})

    # Call the visualization methods to convert the 3D tensors to 2D grayscale.
    vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_mask_3d)

    # Set up matplot lib figures.
    P.figure(figsize=(_ROWS * _UPSCALE_FACTOR, _COLS * _UPSCALE_FACTOR))

    # Render the saliency masks.
    ShowImage(img, title='Original Image (%s)'%(img_class_name[img_class]), ax=P.subplot(_ROWS, _COLS, 1))
    #correct_wrong = '' if pred == img_class else ' - '+img_class_name[pred]
    #ShowGrayscaleImage(vanilla_mask_grayscale, title='Vanilla Gradient%s'%(correct_wrong), ax=P.subplot(_ROWS, _COLS, 2))
    print_str = NN_model.detail + ' - %.2f'%(prob[img_class]) + ('' if pred==img_class else '/%s %.2f'%(img_class_name[pred], prob[pred]))
    ShowGrayscaleImage(vanilla_mask_grayscale, title=print_str, ax=P.subplot(_ROWS, _COLS, 2))


def plot_vanilla_smoothed_gradient_mask(graph, sess, NN_model, img, img_class, img_class_name):
    pred, prob = sess.run([NN_model.pred, NN_model.prob], feed_dict={NN_model.model_input: [img]})
    gradient_saliency = saliency.GradientSaliency(graph, sess, NN_model.y, NN_model.model_input)

    # Get 3D mask
    smoothgrad_mask_3d = gradient_saliency.GetSmoothedMask(img, feed_dict = {NN_model.class_selector: img_class})

    # Call the visualization methods to convert the 3D tensors to 2D grayscale.
    smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_mask_3d)

    # Set up matplot lib figures.
    P.figure(figsize=(_ROWS * _UPSCALE_FACTOR, _COLS * _UPSCALE_FACTOR))

    # Render the saliency masks.
    ShowImage(img, title='Original Image (%s)'%(img_class_name[img_class]), ax=P.subplot(_ROWS, _COLS, 1))
    #correct_wrong = '' if pred == img_class else ' - '+img_class_name[pred]
    #ShowGrayscaleImage(smoothgrad_mask_grayscale, title='SmoothGrad%s'%(correct_wrong), ax=P.subplot(_ROWS, _COLS, 2))
    print_str = NN_model.detail + ' - %.2f'%(prob[img_class]) + ('' if pred==img_class else '/%s %.2f'%(img_class_name[pred], prob[pred]))
    ShowGrayscaleImage(smoothgrad_mask_grayscale, title=print_str, ax=P.subplot(_ROWS, _COLS, 2))


def plot_guided_gradient_mask(graph, sess, NN_model, img, img_class, img_class_name):
    pred, prob = sess.run([NN_model.pred, NN_model.prob], feed_dict={NN_model.model_input: [img]})
    gradient_saliency = saliency.GuidedBackprop(graph, sess, NN_model.y, NN_model.model_input)

    # Get 3D mask
    vanilla_guided_mask_3d = gradient_saliency.GetMask(img, feed_dict = {NN_model.class_selector: img_class})

    # Call the visualization methods to convert the 3D tensors to 2D grayscale.
    vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_guided_mask_3d)

    # Set up matplot lib figures.
    P.figure(figsize=(_ROWS * _UPSCALE_FACTOR, _COLS * _UPSCALE_FACTOR))

    # Render the saliency masks.
    ShowImage(img, title='Original Image (%s)'%(img_class_name[img_class]), ax=P.subplot(_ROWS, _COLS, 1))
    #correct_wrong = '' if pred == img_class else ' - '+img_class_name[pred]
    #ShowGrayscaleImage(vanilla_mask_grayscale, title='Vanilla Guided Backprop%s'%(correct_wrong), ax=P.subplot(_ROWS, _COLS, 2))
    print_str = NN_model.detail + ' - %.2f'%(prob[img_class]) + ('' if pred==img_class else '/%s %.2f'%(img_class_name[pred], prob[pred]))
    ShowGrayscaleImage(vanilla_mask_grayscale, title=print_str, ax=P.subplot(_ROWS, _COLS, 2))


def plot_guided_smoothed_gradient_mask(graph, sess, NN_model, img, img_class, img_class_name):
    pred, prob = sess.run([NN_model.pred, NN_model.prob], feed_dict={NN_model.model_input: [img]})
    gradient_saliency = saliency.GuidedBackprop(graph, sess, NN_model.y, NN_model.model_input)

    # Get 3D mask
    smoothgrad_guided_mask_3d = gradient_saliency.GetSmoothedMask(img, feed_dict = {NN_model.class_selector: img_class})

    # Call the visualization methods to convert the 3D tensors to 2D grayscale.
    smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_guided_mask_3d)

    # Set up matplot lib figures.
    P.figure(figsize=(_ROWS * _UPSCALE_FACTOR, _COLS * _UPSCALE_FACTOR))

    # Render the saliency masks.
    ShowImage(img, title='Original Image (%s)'%(img_class_name[img_class]), ax=P.subplot(_ROWS, _COLS, 1))
    #correct_wrong = '' if pred == img_class else ' - '+img_class_name[pred]
    #ShowGrayscaleImage(smoothgrad_mask_grayscale, title='SmoothGrad Guided Backprop%s'%(correct_wrong), ax=P.subplot(_ROWS, _COLS, 2))
    print_str = NN_model.detail + ' - %.2f'%(prob[img_class]) + ('' if pred==img_class else '/%s %.2f'%(img_class_name[pred], prob[pred]))
    ShowGrayscaleImage(smoothgrad_mask_grayscale, title=print_str, ax=P.subplot(_ROWS, _COLS, 2))


def plot_integrated_gradient_mask(graph, sess, NN_model, img, img_class, img_class_name):
    pred, prob = sess.run([NN_model.pred, NN_model.prob], feed_dict={NN_model.model_input: [img]})
    gradient_saliency = saliency.IntegratedGradients(graph, sess, NN_model.y, NN_model.model_input)

    # Baseline is a black image.
    baseline = np.zeros(img.shape)
    baseline.fill(-1)

    # Get 3D mask
    vanilla_integrated_mask_3d = gradient_saliency.GetMask(img, feed_dict = {NN_model.class_selector: img_class}, x_steps=25, x_baseline=baseline)

    # Call the visualization methods to convert the 3D tensors to 2D grayscale.
    vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_integrated_mask_3d)

    # Set up matplot lib figures.
    P.figure(figsize=(_ROWS * _UPSCALE_FACTOR, _COLS * _UPSCALE_FACTOR))

    # Render the saliency masks.
    ShowImage(img, title='Original Image (%s)'%(img_class_name[img_class]), ax=P.subplot(_ROWS, _COLS, 1))
    #correct_wrong = '' if pred == img_class else ' - '+img_class_name[pred]
    #ShowGrayscaleImage(vanilla_mask_grayscale, title='Vanilla Integrated Gradients%s'%(correct_wrong), ax=P.subplot(_ROWS, _COLS, 2))
    print_str = NN_model.detail + ' - %.2f'%(prob[img_class]) + ('' if pred==img_class else '/%s %.2f'%(img_class_name[pred], prob[pred]))
    ShowGrayscaleImage(vanilla_mask_grayscale, title=print_str, ax=P.subplot(_ROWS, _COLS, 2))


def plot_integrated_smoothed_gradient_mask(graph, sess, NN_model, img, img_class, img_class_name):
    pred, prob = sess.run([NN_model.pred, NN_model.prob], feed_dict={NN_model.model_input: [img]})
    gradient_saliency = saliency.IntegratedGradients(graph, sess, NN_model.y, NN_model.model_input)

    # Baseline is a black image.
    baseline = np.zeros(img.shape)
    baseline.fill(-1)

    # Get 3D mask
    smoothgrad_integrated_mask_3d = gradient_saliency.GetSmoothedMask(img, feed_dict = {NN_model.class_selector: img_class}, x_steps=25, x_baseline=baseline)

    # Call the visualization methods to convert the 3D tensors to 2D grayscale.
    smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_integrated_mask_3d)

    # Set up matplot lib figures.
    P.figure(figsize=(_ROWS * _UPSCALE_FACTOR, _COLS * _UPSCALE_FACTOR))

    # Render the saliency masks.
    ShowImage(img, title='Original Image (%s)'%(img_class_name[img_class]), ax=P.subplot(_ROWS, _COLS, 1))
    #correct_wrong = '' if pred == img_class else ' - '+img_class_name[pred]
    #ShowGrayscaleImage(smoothgrad_mask_grayscale, title='Smoothgrad Integrated Gradients%s'%(correct_wrong), ax=P.subplot(_ROWS, _COLS, 2))
    print_str = NN_model.detail + ' - %.2f'%(prob[img_class]) + ('' if pred==img_class else '/%s %.2f'%(img_class_name[pred], prob[pred]))
    ShowGrayscaleImage(smoothgrad_mask_grayscale, title=print_str, ax=P.subplot(_ROWS, _COLS, 2))


###################################################################################################################
###################################################################################################################
#########     Plot functions with list of models
###################################################################################################################
###################################################################################################################
#_multiROWS, _multiCOLS = 3, 4
#_plot_loc = [1, 1+_multiCOLS, 1+2*_multiCOLS, 2, 2+_multiCOLS, 3, 3+_multiCOLS, 3+2*_multiCOLS, 4, 4+_multiCOLS, 4+2*_multiCOLS]
_multiROWS, _multiCOLS = 3, 5
_plot_loc = [1, 2, 2+2*_multiCOLS, 3, 3+_multiCOLS, 4, 4+_multiCOLS, 4+2*_multiCOLS, 5, 5+_multiCOLS, 5+2*_multiCOLS]

def plot_vanilla_gradient_masks(graph, sess, NN_models, models_list, img, img_class, img_class_names):
    mask_grayscales, pred_list, prob_list = [img], [img_class], [[]]
    for model_to_plot in models_list:
        NN_model = NN_models[model_to_plot]
        pred_list.append(sess.run(NN_model.pred, feed_dict={NN_model.model_input: [img]}))
        prob_list.append(sess.run(NN_model.prob, feed_dict={NN_model.model_input: [img]}))
        gradient_saliency = saliency.GradientSaliency(graph, sess, NN_model.y, NN_model.model_input)

        # Get 3D mask
        vanilla_mask_3d = gradient_saliency.GetMask(img, feed_dict = {NN_model.class_selector: img_class})

        # Call the visualization methods to convert the 3D tensors to 2D grayscale.
        vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_mask_3d)

        mask_grayscales.append(vanilla_mask_grayscale)

    models_list.insert(0, 'Original Image')
    pred_conds = [None] + [pred_list[0]==pred_list[cnt] for cnt in range(1, len(pred_list))]
    # Set up matplot lib figures.
    P.figure(figsize=(_multiROWS * _UPSCALE_FACTOR, _multiCOLS * _UPSCALE_FACTOR))

    # Render the saliency masks.
    for img_cnt, (img_to_plot, model_to_plot, pred_cond, pred, prob) in enumerate(zip(mask_grayscales, models_list, pred_conds, pred_list, prob_list)):
        if img_cnt == 0:
            ShowImage(img_to_plot, title='Original Image (%s)'%(img_class_names[img_class]), ax=P.subplot(_multiROWS, _multiCOLS, _plot_loc[img_cnt]))
        else:
            #correct_wrong = 'c' if pred_cond else 'w\n'+img_class_names[pred]
            #ShowGrayscaleImage(img_to_plot, title='Vanilla - %s/%s'%(model_to_plot, correct_wrong), ax=P.subplot(_multiROWS, _multiCOLS, _plot_loc[img_cnt]))
            print_str = '%s - %.2f' %(model_to_plot, prob[img_class]) + ('' if pred_cond else '/%s %.2f'%(img_class_names[pred], prob[pred]))
            ShowGrayscaleImage(img_to_plot, title=print_str, ax=P.subplot(_multiROWS, _multiCOLS, _plot_loc[img_cnt]))


def plot_vanilla_smoothed_gradient_masks(graph, sess, NN_models, models_list, img, img_class, img_class_names):
    mask_grayscales, pred_list, prob_list = [img], [img_class], [[]]
    for model_to_plot in models_list:
        NN_model = NN_models[model_to_plot]
        pred_list.append(sess.run(NN_model.pred, feed_dict={NN_model.model_input: [img]}))
        prob_list.append(sess.run(NN_model.prob, feed_dict={NN_model.model_input: [img]}))
        gradient_saliency = saliency.GradientSaliency(graph, sess, NN_model.y, NN_model.model_input)

        # Get 3D mask
        smoothgrad_mask_3d = gradient_saliency.GetSmoothedMask(img, feed_dict = {NN_model.class_selector: img_class})

        # Call the visualization methods to convert the 3D tensors to 2D grayscale.
        smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_mask_3d)

        mask_grayscales.append(smoothgrad_mask_grayscale)

    models_list.insert(0, 'Original Image')
    pred_conds = [None] + [pred_list[0]==pred_list[cnt] for cnt in range(1, len(pred_list))]
    # Set up matplot lib figures.
    P.figure(figsize=(_multiROWS * _UPSCALE_FACTOR, _multiCOLS * _UPSCALE_FACTOR))

    # Render the saliency masks.
    for img_cnt, (img_to_plot, model_to_plot, pred_cond, pred, prob) in enumerate(zip(mask_grayscales, models_list, pred_conds, pred_list, prob_list)):
        if img_cnt == 0:
            ShowImage(img_to_plot, title='Original Image (%s)'%(img_class_names[img_class]), ax=P.subplot(_multiROWS, _multiCOLS, _plot_loc[img_cnt]))
        else:
            #correct_wrong = 'c' if pred_cond else 'w\n'+img_class_names[pred]
            #ShowGrayscaleImage(img_to_plot, title='Smooth - %s/%s'%(model_to_plot, correct_wrong), ax=P.subplot(_multiROWS, _multiCOLS, _plot_loc[img_cnt]))
            print_str = '%s - %.2f' %(model_to_plot, prob[img_class]) + ('' if pred_cond else '/%s %.2f'%(img_class_names[pred], prob[pred]))
            ShowGrayscaleImage(img_to_plot, title=print_str, ax=P.subplot(_multiROWS, _multiCOLS, _plot_loc[img_cnt]))


def plot_guided_gradient_masks(graph, sess, NN_models, models_list, img, img_class, img_class_names):
    mask_grayscales, pred_list, prob_list = [img], [img_class], [[]]
    for model_to_plot in models_list:
        NN_model = NN_models[model_to_plot]
        pred_list.append(sess.run(NN_model.pred, feed_dict={NN_model.model_input: [img]}))
        prob_list.append(sess.run(NN_model.prob, feed_dict={NN_model.model_input: [img]}))
        gradient_saliency = saliency.GuidedBackprop(graph, sess, NN_model.y, NN_model.model_input)

        # Get 3D mask
        vanilla_guided_mask_3d = gradient_saliency.GetMask(img, feed_dict = {NN_model.class_selector: img_class})

        # Call the visualization methods to convert the 3D tensors to 2D grayscale.
        vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_guided_mask_3d)

        mask_grayscales.append(vanilla_mask_grayscale)

    models_list.insert(0, 'Original Image')
    pred_conds = [None] + [pred_list[0]==pred_list[cnt] for cnt in range(1, len(pred_list))]
    # Set up matplot lib figures.
    P.figure(figsize=(_multiROWS * _UPSCALE_FACTOR, _multiCOLS * _UPSCALE_FACTOR))

    # Render the saliency masks.
    for img_cnt, (img_to_plot, model_to_plot, pred_cond, pred, prob) in enumerate(zip(mask_grayscales, models_list, pred_conds, pred_list, prob_list)):
        if img_cnt == 0:
            ShowImage(img_to_plot, title='Original Image (%s)'%(img_class_names[img_class]), ax=P.subplot(_multiROWS, _multiCOLS, _plot_loc[img_cnt]))
        else:
            #correct_wrong = 'c' if pred_cond else 'w\n'+img_class_names[pred]
            #ShowGrayscaleImage(img_to_plot, title='Vanilla Guided - %s/%s'%(model_to_plot, correct_wrong), ax=P.subplot(_multiROWS, _multiCOLS, _plot_loc[img_cnt]))
            print_str = '%s - %.2f' %(model_to_plot, prob[img_class]) + ('' if pred_cond else '/%s %.2f'%(img_class_names[pred], prob[pred]))
            ShowGrayscaleImage(img_to_plot, title=print_str, ax=P.subplot(_multiROWS, _multiCOLS, _plot_loc[img_cnt]))


def plot_guided_smoothed_gradient_masks(graph, sess, NN_models, models_list, img, img_class, img_class_names):
    masks, mask_grayscales, pred_list, prob_list = [], [img], [img_class], [[]]
    for model_to_plot in models_list:
        NN_model = NN_models[model_to_plot]
        pred_list.append(sess.run(NN_model.pred, feed_dict={NN_model.model_input: [img]}))
        prob_list.append(sess.run(NN_model.prob, feed_dict={NN_model.model_input: [img]}))
        gradient_saliency = saliency.GuidedBackprop(graph, sess, NN_model.y, NN_model.model_input)

        # Get 3D mask
        smoothgrad_guided_mask_3d = gradient_saliency.GetSmoothedMask(img, feed_dict = {NN_model.class_selector: img_class})

        # Call the visualization methods to convert the 3D tensors to 2D grayscale.
        smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_guided_mask_3d)

        masks.append(np.sum(np.abs(smoothgrad_guided_mask_3d), axis=2))
        mask_grayscales.append(smoothgrad_mask_grayscale)

    models_list.insert(0, 'Original Image')
    pred_conds = [None] + [pred_list[0]==pred_list[cnt] for cnt in range(1, len(pred_list))]

    masks.insert(0, np.zeros_like(masks[-1]))
    masks = np.array(masks)

    # Set up matplot lib figures.
    fig = P.figure(figsize=(_multiROWS * _UPSCALE_FACTOR, _multiCOLS * _UPSCALE_FACTOR))

    # Render the saliency masks.
    #for img_cnt, (img_to_plot, model_to_plot, pred_cond, pred, prob) in enumerate(zip(mask_grayscales, models_list, pred_conds, pred_list, prob_list)):
    for img_cnt, (img_to_plot, mask, model_to_plot, pred_cond, pred, prob) in enumerate(zip(mask_grayscales, masks, models_list, pred_conds, pred_list, prob_list)):
        if img_cnt == 0:
            ShowImage(img_to_plot, title='Original Image (%s)'%(img_class_names[img_class]), ax=P.subplot(_multiROWS, _multiCOLS, _plot_loc[img_cnt]))
        else:
            #correct_wrong = 'c' if pred_cond else 'w\n'+img_class_names[pred]
            #ShowGrayscaleImage(img_to_plot, title='Smooth Guided - %s/%s'%(model_to_plot, correct_wrong), ax=P.subplot(_multiROWS, _multiCOLS, _plot_loc[img_cnt]))
            print_str = '%s - %.2f' %(model_to_plot, prob[img_class]) + ('' if pred_cond else '/%s %.2f'%(img_class_names[pred], prob[pred]))
            ShowGrayscaleImage(img_to_plot, title=print_str, ax=P.subplot(_multiROWS, _multiCOLS, _plot_loc[img_cnt]))
            #ax = fig.add_subplot(_multiROWS, _multiCOLS, _plot_loc[img_cnt])
            #P.imshow(mask, cmap='viridis', vmin=masks.min(), vmax=masks.max())
            #P.title(print_str)
            #ax.axis('off')


def plot_integrated_gradient_masks(graph, sess, NN_models, models_list, img, img_class, img_class_names):
    mask_grayscales, pred_list, prob_list = [img], [img_class], [[]]
    for model_to_plot in models_list:
        NN_model = NN_models[model_to_plot]
        pred_list.append(sess.run(NN_model.pred, feed_dict={NN_model.model_input: [img]}))
        prob_list.append(sess.run(NN_model.prob, feed_dict={NN_model.model_input: [img]}))
        gradient_saliency = saliency.IntegratedGradients(graph, sess, NN_model.y, NN_model.model_input)

        # Baseline is a black image.
        baseline = np.zeros(img.shape)
        baseline.fill(-1)

        # Get 3D mask
        vanilla_integrated_mask_3d = gradient_saliency.GetMask(img, feed_dict = {NN_model.class_selector: img_class}, x_steps=25, x_baseline=baseline)

        # Call the visualization methods to convert the 3D tensors to 2D grayscale.
        vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_integrated_mask_3d)

        mask_grayscales.append(vanilla_mask_grayscale)

    models_list.insert(0, 'Original Image')
    pred_conds = [None] + [pred_list[0]==pred_list[cnt] for cnt in range(1, len(pred_list))]
    # Set up matplot lib figures.
    P.figure(figsize=(_multiROWS * _UPSCALE_FACTOR, _multiCOLS * _UPSCALE_FACTOR))

    # Render the saliency masks.
    for img_cnt, (img_to_plot, model_to_plot, pred_cond, pred, prob) in enumerate(zip(mask_grayscales, models_list, pred_conds, pred_list, prob_list)):
        if img_cnt == 0:
            ShowImage(img_to_plot, title='Original Image (%s)'%(img_class_names[img_class]), ax=P.subplot(_multiROWS, _multiCOLS, _plot_loc[img_cnt]))
        else:
            #correct_wrong = 'c' if pred_cond else 'w\n'+img_class_names[pred]
            #ShowGrayscaleImage(img_to_plot, title='Vanilla Integrated - %s/%s'%(model_to_plot, correct_wrong), ax=P.subplot(_multiROWS, _multiCOLS, _plot_loc[img_cnt]))
            print_str = '%s - %.2f' %(model_to_plot, prob[img_class]) + ('' if pred_cond else '/%s %.2f'%(img_class_names[pred], prob[pred]))
            ShowGrayscaleImage(img_to_plot, title=print_str, ax=P.subplot(_multiROWS, _multiCOLS, _plot_loc[img_cnt]))


def plot_integrated_smoothed_gradient_masks(graph, sess, NN_models, models_list, img, img_class, img_class_names):
    mask_grayscales, pred_list, prob_list = [img], [img_class], [[]]
    for model_to_plot in models_list:
        NN_model = NN_models[model_to_plot]
        pred_list.append(sess.run(NN_model.pred, feed_dict={NN_model.model_input: [img]}))
        prob_list.append(sess.run(NN_model.prob, feed_dict={NN_model.model_input: [img]}))
        gradient_saliency = saliency.IntegratedGradients(graph, sess, NN_model.y, NN_model.model_input)

        # Baseline is a black image.
        baseline = np.zeros(img.shape)
        baseline.fill(-1)

        # Get 3D mask
        smoothgrad_integrated_mask_3d = gradient_saliency.GetSmoothedMask(img, feed_dict = {NN_model.class_selector: img_class}, x_steps=25, x_baseline=baseline)

        # Call the visualization methods to convert the 3D tensors to 2D grayscale.
        smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_integrated_mask_3d)

        mask_grayscales.append(smoothgrad_mask_grayscale)

    models_list.insert(0, 'Original Image')
    pred_conds = [None] + [pred_list[0]==pred_list[cnt] for cnt in range(1, len(pred_list))]
    # Set up matplot lib figures.
    P.figure(figsize=(_multiROWS * _UPSCALE_FACTOR, _multiCOLS * _UPSCALE_FACTOR))

    # Render the saliency masks.
    for img_cnt, (img_to_plot, model_to_plot, pred_cond, pred, prob) in enumerate(zip(mask_grayscales, models_list, pred_conds, pred_list, prob_list)):
        if img_cnt == 0:
            ShowImage(img_to_plot, title='Original Image (%s)'%(img_class_names[img_class]), ax=P.subplot(_multiROWS, _multiCOLS, _plot_loc[img_cnt]))
        else:
            #correct_wrong = 'c' if pred_cond else 'w\n'+img_class_names[pred]
            #ShowGrayscaleImage(img_to_plot, title='Smooth Integrated - %s/%s'%(model_to_plot, correct_wrong), ax=P.subplot(_multiROWS, _multiCOLS, _plot_loc[img_cnt]))
            print_str = '%s - %.2f' %(model_to_plot, prob[img_class]) + ('' if pred_cond else '/%s %.2f'%(img_class_names[pred], prob[pred]))
            ShowGrayscaleImage(img_to_plot, title=print_str, ax=P.subplot(_multiROWS, _multiCOLS, _plot_loc[img_cnt]))