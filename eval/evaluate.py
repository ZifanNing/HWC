import numpy as np
from sklearn import manifold
import torch
import visual.visdom
import visual.plt
import utils

####--------------------------------------------------------------------------------------------------------------####

####-----------------------------####
####----CLASSIFIER EVALUATION----####
####-----------------------------####

def validate(model, dataset, batch_size=128, test_size=1024, verbose=True, allowed_classes=None, no_task_mask=False, task=None):
    '''Evaluate precision (= accuracy or proportion correct) of a classifier ([model]) on [dataset].

    [allowed_classes]   None or <list> containing all 'active classes' between which should be chosen
                            (these 'active classes' are assumed to be contiguous)'''

    # Get device-type / using cuda?
    device = model._device()
    cuda = model._is_on_cuda()

    # Set model to eval()-mode
    model.eval()

    # Apply task-specifc 'gating-mask' for each hidden fully connected layer (or remove it!)
    if model.mask_dict is not None:
        if no_task_mask:
            model.reset_XdGmask()
        else:
            model.apply_XdGmask(task=task)

    # Loop over batches in [dataset]
    data_loader = utils.get_data_loader(dataset, batch_size, cuda=cuda)
    total_tested = total_correct = 0
    for data, labels in data_loader:
        # -break on [test_size] (if 'None', full dataset is used)
        if test_size:
            if total_tested >= test_size:
                break
        # -evaluate model (if requested, only on [allowed_classes])
        data, labels = data.to(device), labels.to(device)
        labels = labels
        with torch.no_grad():
            scores = model.classify(data, not_hidden=True)
            if type(scores) == list:
                spikes = 0
                for i in range(len(scores)):
                    spikes += scores[i]
                scores = spikes / len(scores)
            scores = scores if (allowed_classes is None) else scores[:, allowed_classes]
            _, predicted = torch.max(scores, 1)
        # -update statistics
        total_correct += (predicted == labels).sum().item()
        total_tested += len(data)
    precision = total_correct / total_tested
    if allowed_classes is not None:
        print(allowed_classes, ':', precision)
    # else:
    #     print('accuracy:', precision)

    # Print result on screen (if requested) and return it
    if verbose:
        print('=> precision: {:.3f}'.format(precision))

    return precision

def initiate_precision_dict(n_tasks):
    '''Initiate <dict> with all precision-measures to keep track of.'''
    precision = {}
    precision['all_tasks'] = [[] for _ in range(n_tasks)]
    precision['average'] = []
    precision['x_iteration'] = []
    precision['x_task'] = []

    return precision

def precision(model, datasets, current_task, iteration, classes_per_task=None, scenario='none', precision_dict=None, test_size=None, visdom=None, verbose=False, no_task_mask=False):
    '''Evaluate precision of a classifier (=[model]) on all tasks so far (= up to [current_task]) using [datasets].

    [precision_dict]    None or <dict> of all measures to keep track of, to which results will be appended to
    [classes_per_task]  <int> number of active classes er task
    [scenario]          <str> how to decide which classes to include during evaluating precision
    [visdom]            None or <dict> with name of 'graph' and 'env' (if None, no visdom-plots are made)'''

    # Evaluate accuracy of model predictions for all tasks so far (reporting '0' for future tasks)
    n_tasks = len(datasets)
    precs = []

    if model.experiment == 'expandMNIST':
        task_label = [list(range(j, j + classes_per_task)) for j in range(n_tasks)]
    elif model.experiment == 'singleMNIST':
        task_label = [list(range(int((j // 2) * 2), int(((j // 2) + 1) * 2))) for j in range(n_tasks)]
    else:
        task_label = [list(range(classes_per_task * j, classes_per_task * (j + 1))) for j in range(n_tasks)]

    for i in range(n_tasks):
        if i + 1 <= current_task:
            if scenario == 'task':
                if model.change_order:
                    allowed_classes = task_label[model.task_order[i]]
                else:
                    allowed_classes = task_label[i]

            elif scenario == 'class':
                allowed_classes = list(range(current_task - 1 + classes_per_task)) if model.experiment == 'expandMNIST' else list(range(classes_per_task * (current_task)))
            else:
                allowed_classes = None
            precs.append(validate(model, datasets[i], test_size=test_size, verbose=verbose, allowed_classes=allowed_classes, no_task_mask=no_task_mask, task=i + 1))
        else:
            precs.append(0)
    if scenario == 'domain':
        print(precs)
    average_precs = sum([precs[task_id] if task_id == 0 else precs[task_id] for task_id in range(current_task)]) / (current_task)
    print('average:', average_precs)

    # Print results on screen
    if verbose:
        print(' => ave precision: {:.3f}'.format(average_precs))

    # Send results to visdom server
    names = ['task {}'.format(i + 1) for i in range(n_tasks)]
    if visdom is not None:
        visual.visdom.visualize_scalars(scalars=precs, names=names, iteration=iteration, title='Accuracy per task ({})'.format(visdom['graph']), env=visdom['env'], ylabel='precision')
        if n_tasks > 1:
            visual.visdom.visualize_scalars(scalars=[average_precs], names=['ave precision'], iteration=iteration, title='Average accuracy ({})'.format(visdom['graph']), env=visdom['env'], ylabel='precision')

    # Append results to [progress]-dictionary and return
    if precision_dict is not None:
        for task_id, _ in enumerate(names):
            precision_dict['all_tasks'][task_id].append(precs[task_id])
        precision_dict['average'].append(average_precs)
        precision_dict['x_iteration'].append(iteration)
        precision_dict['x_task'].append(current_task)

    return precision_dict

####--------------------------------------------------------------------------------------------------------------####

####------------------------------------------####
####----VISUALIZE EXTRACTED REPRESENTATION----####
####------------------------------------------####

def visualize_latent_space(model, X, y=None, visdom=None, pdf=None, verbose=False):
    '''Show T-sne projection of feature representation used to classify from (with each class in different color).'''

    # Set model to eval()-mode
    model.eval()

    # Compute the representation used for classification
    if verbose:
        print('Computing feature space...')
    with torch.no_grad():
        z_mean = model.feature_extractor(X)

    # Compute t-SNE embedding of latent space (unless z has 2 dimensions!)
    if z_mean.size()[1] == 2:
        z_tsne = z_mean.cpu().numpy()
    else:
        if verbose:
            print('Computing t-SNE embedding...')
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
        z_tsne = tsne.fit_transform(z_mean.cpu())

    # Plot images according to t-sne embedding
    if pdf is not None:
        figure = visual.plt.plot_scatter(z_tsne[:, 0], z_tsne[:, 1], colors=y)
        pdf.savefig(figure)
    if visdom is not None:
        message = ('Visualization of extracted representation')
        visual.visdom.scatter_plot(z_tsne, title='{} ({})'.format(message, visdom['graph']), colors=y+1 if y is not None else y, env=visdom['env'])

####--------------------------------------------------------------------------------------------------------------####

####----------------------------####
####----GENERATOR EVALUATION----####
####----------------------------####

def show_samples(model, config, pdf=None, visdom=None, size=32, sample_mode=None, title='Generated samples', allowed_classes=None, allowed_domains=None):
    '''Plot samples from a generative model in [pdf] and/or in [visdom].'''

    # Set model to evaluation-mode
    model.eval()

    # Generate samples from the model
    sample = model.sample(size, sample_mode=sample_mode, allowed_classes=allowed_classes, allowed_domains=allowed_domains, only_x=True)
    # -correctly arrange pixel-values and move to cpu (if needed)
    image_tensor = sample.view(-1, config['channels'], config['size'], config['size']).cpu()
    # -denormalize images if needed
    if config['normalize']:
        image_tensor = config['denormalize'](image_tensor).clamp(min=0, max=1)

    # Plot generated images in [pdf] and/or [visdom]
    # -number of rows
    nrow = int(np.ceil(np.sqrt(size)))
    # -make plots
    if pdf is not None:
        visual.plt.plot_images_from_tensor(image_tensor, pdf, title=title, nrow=nrow)
    if visdom is not None:
        mode = '' if sample_mode is None else '(mode = {})'.format(sample_mode)
        visual.visdom.visualize_images(tensor=image_tensor, env=visdom['env'], nrow=nrow, title='Generated samples {} ({})'.format(mode, visdom['graph']),)

####--------------------------------------------------------------------------------------------------------------####

####--------------------------------####
####----RECONSTRUCTOR EVALUATION----####
####--------------------------------####

def show_reconstruction(model, dataset, config, pdf=None, visdom=None, size=32, epoch=None, task=None, no_task_mask=False):
    '''Plot reconstructed examples by an auto-encoder [model] on [dataset], either in [pdf] and/or in [visdom].'''

    # Get device-type / using cuda?
    cuda = model._is_on_cuda()
    device = model._device()

    # Set model to evaluation-mode
    model.eval()

    # Get data
    data_loader = utils.get_data_loader(dataset, size, cuda=cuda)
    (data, labels) = next(iter(data_loader))

    # If needed, apply correct specific task-mask (for fully-connected hidden layers in encoder)
    if hasattr(model, 'mask_dict') and model.mask_dict is not None:
        if no_task_mask:
            model.reset_XdGmask()
        else:
            model.apply_XdGmask(task=task)

    # Evaluate model
    data, labels = data.to(device), labels.to(device)
    with torch.no_grad():
        gate_input = (torch.tensor(np.repeat(task - 1, size)).to(device) if model.dg_type == 'task' else labels) if (utils.checkattr(model, 'dg_gates') and model.dg_prop > 0) else None
        recon_output = model(data, gate_input=gate_input, full=True, reparameterize=False)
    recon_batch = recon_output[0]

    # Plot original and reconstructed images
    # -number of rows
    nrow = int(np.ceil(np.sqrt(size * 2)))
    # -collect and arrange pixel-values
    comparison = torch.cat([data.view(-1, config['channels'], config['size'], config['size'])[:size], recon_batch.view(-1, config['channels'], config['size'], config['size'])[:size]]).cpu()
    image_tensor = comparison.view(-1, config['channels'], config['size'], config['size'])
    # -denormalize images if needed
    if config['normalize']:
        image_tensor = config['denormalize'](image_tensor).clamp(min=0, max=1)
    # -make plots
    if pdf is not None:
        epoch_stm = '' if epoch is None else ' after epoch '.format(epoch)
        task_stm = '' if task is None else ' (task {})'.format(task)
        visual.plt.plot_images_from_tensor(image_tensor, pdf, nrow=nrow, title='Reconstructions' + task_stm + epoch_stm)
    if visdom is not None:
        visual.visdom.visualize_images(tensor=image_tensor, title='Reconstructions ({})'.format(visdom['graph']), env=visdom['env'], nrow=nrow,)