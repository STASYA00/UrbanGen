def crop(image, new_shape):
	'''
	Function for cropping an image tensor: Given an image tensor and the new shape,
	crops to the center pixels (assumes that the input's size and the new size are
	even numbers).
	Parameters:
		image: image tensor of shape (batch size, channels, height, width)
		new_shape: a torch.Size object with the shape you want x to have
	'''
	middle_height = image.shape[2] // 2
	middle_width = image.shape[3] // 2
	starting_height = middle_height - new_shape[2] // 2
	final_height = starting_height + new_shape[2]
	starting_width = middle_width - new_shape[3] // 2
	final_width = starting_width + new_shape[3]
	cropped_image = image[:, :, starting_height:final_height, starting_width:final_width]
	return cropped_image

def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', use_sigmoid=False,
             init_type='normal', init_gain=0.02, gpu_id='cuda:0'):
	net = None
	norm_layer = get_norm_layer(norm_type=norm)

	if netD == 'basic':
		net = NLayerDiscriminator(input_nc, ndf, n_layers=3,
		                          norm_layer=norm_layer,
		                          use_sigmoid=use_sigmoid)
	elif netD == 'n_layers':
		net = NLayerDiscriminator(input_nc, ndf, n_layers_D,
		                          norm_layer=norm_layer,
		                          use_sigmoid=use_sigmoid)
	elif netD == 'pixel':
		net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer,
		                         use_sigmoid=use_sigmoid)
	elif netD == 'activation':
		net = ActivationDiscriminator(input_nc, ndf, n_layers_D,
		                              norm_layer=norm_layer,
		                              use_sigmoid=use_sigmoid)
	elif netD == 'classification':
		net = Classifier(input_nc, len(buildings))
	else:
		raise NotImplementedError(
			'Discriminator model name [%s] is not recognized' % net)

	return init_net(net, init_type, init_gain, gpu_id)


def define_G(input_nc, output_nc, ngf, norm='batch', use_dropout=False,
			 init_type='normal', init_gain=0.02, gpu_id='cuda:0'):
	net = None
	norm_layer = get_norm_layer(norm_type=norm)

	net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer,
						  use_dropout=use_dropout, n_blocks=9)

	return init_net(net, init_type, init_gain, gpu_id)


def get_gen_class_loss(gen, disc, real, condition, adv_criterion,
                       recon_criterion, lambda_recon):
	fake = gen(condition)
	ev = disc(fake, condition[:, :3, :, :])
	adv_loss = adv_criterion(ev, torch.ones(ev.shape).to(device))
	rec_loss = recon_criterion(fake, real)
	gen_loss = torch.sum(adv_loss) + (rec_loss * lambda_recon)
	return gen_loss


def get_gen_loss(gen, disc, real, condition, adv_criterion, recon_criterion,
                 lambda_recon):
	fake = gen(condition)
	ev = disc(fake, condition)
	adv_loss = adv_criterion(ev, torch.ones(ev.shape).to(device))
	rec_loss = recon_criterion(fake, real)
	gen_loss = torch.sum(adv_loss) + (rec_loss * lambda_recon
	return gen_loss


def get_norm_layer(norm_type='instance'):
	if norm_type == 'batch':
		norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
	elif norm_type == 'instance':
		norm_layer = functools.partial(nn.InstanceNorm2d, affine=False,
									   track_running_stats=False)
	elif norm_type == 'switchable':
		norm_layer = SwitchNorm2d
	elif norm_type == 'none':
		norm_layer = None
	else:
		raise NotImplementedError(
			'normalization layer [%s] is not found' % norm_type)
	return norm_layer


def get_scheduler(optimizer):
	if lr_policy == 'lambda':
		def lambda_rule(epoch):
			lr_l = 1.0 - max(0, epoch + epoch_count - niter) / float(
				niter_decay + 1)
			print('New lr: {}'.format(lr_l))
			return lr_l

		scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
	elif lr_policy == 'step':
		scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_iters,
										gamma=0.1)
	elif lr_policy == 'plateau':
		scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
												   factor=0.2, threshold=0.01,
												   patience=5)
	elif lr_policy == 'cosine':
		scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=niter,
												   eta_min=0)
	else:
		return NotImplementedError(
			'learning rate policy [%s] is not implemented', lr_policy)
	return scheduler


def init_net(net, init_type='normal', init_gain=0.02, gpu_id='cuda:0'):
	net.to(gpu_id)
	init_weights(net, init_type, gain=init_gain)
	return net


def init_weights(net, init_type='normal', gain=0.02):
	def init_func(m):
		classname = m.__class__.__name__
		if hasattr(m, 'weight') and (
				classname.find('Conv') != -1 or classname.find('Linear') != -1):
			if init_type == 'normal':
				nn.init.normal_(m.weight.data, 0.0, gain)
			elif init_type == 'xavier':
				nn.init.xavier_normal_(m.weight.data, gain=gain)
			elif init_type == 'kaiming':
				nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
			elif init_type == 'orthogonal':
				nn.init.orthogonal_(m.weight.data, gain=gain)
			else:
				raise NotImplementedError(
					'initialization method [%s] is not implemented' % init_type)
			if hasattr(m, 'bias') and m.bias is not None:
				nn.init.constant_(m.bias.data, 0.0)
		elif classname.find('BatchNorm2d') != -1:
			nn.init.normal_(m.weight.data, 1.0, gain)
			nn.init.constant_(m.bias.data, 0.0)

	print('initialize network with %s' % init_type)
	net.apply(init_func)


def show_image(tensor):
	_tensor = (tensor + 1) / 2
	_tensor = _tensor.detach().cpu()[0]
	_tensor = _tensor.permute(1, 2, 0).squeeze()
	plt.imshow(_tensor.numpy())


def save_image(tensor, name):
	_tensor = (tensor + 1) / 2
	_tensor = _tensor.detach().cpu()[0]
	_tensor = _tensor.permute(1, 2, 0).squeeze()
	plt.imsave(name + '.png', _tensor.numpy())


def update_learning_rate(scheduler, optimizer):
	scheduler.step()
	lr = optimizer.param_groups[0]['lr']
	print('learning rate = %.7f' % lr)


def weights_init(m):
	if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
		torch.nn.init.normal_(m.weight, 0.0, 0.02)
	if isinstance(m, nn.BatchNorm2d):
		torch.nn.init.normal_(m.weight, 0.0, 0.02)
		torch.nn.init.constant_(m.bias, 0)




