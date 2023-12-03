import torch
import torch.nn as nn
from src.data import SlidingWindowDataLoader, SlidingWindowDataset
from torch.nn import TransformerEncoder
from torch.nn import TransformerDecoder
from src.dlutils import *
from data_config import *
torch.manual_seed(1)

# Proposed Model + Self Conditioning + Adversarial + MAML (VLDB 22)
class TranAD(nn.Module):
	def __init__(self, feats):
		super(TranAD, self).__init__()
		self.name = 'TranAD'
		self.lr = global_learning_rate
		self.batch = global_batch_size
		self.n_feats = feats
		self.n_window = global_window_size
		self.n = self.n_feats * self.n_window
		self.layer_num = global_layer_num
		self.tran_dim = global_tran_dim
		self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
		self.encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=self.tran_dim, dropout=0.1)
		self.transformer_encoder = TransformerEncoder(self.encoder_layers, self.layer_num)
		self.decoder_layers1 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=self.tran_dim, dropout=0.1)
		self.transformer_decoder1 = TransformerDecoder(self.decoder_layers1, self.layer_num)
		self.decoder_layers2 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=self.tran_dim, dropout=0.1)
		self.transformer_decoder2 = TransformerDecoder(self.decoder_layers2, self.layer_num)
		self.fcn1 = nn.Sequential(
			nn.Linear(2 * feats, 2 * feats),
			nn.ReLU(),
			nn.Linear(2 * feats, feats),
			)
		self.fcn2 = nn.Sequential(
			nn.Linear(2 * feats, 2 * feats),
			nn.ReLU(),
			nn.Linear(2 * feats, feats),
			)
		self.fcn3 = nn.Sequential(
			nn.Linear(2 * feats, 2 * feats),
			nn.ReLU(),
			nn.Linear(2 * feats, feats),
			)						

	def encode(self, src, c, tgt):
		src = torch.cat((src, c), dim=2)
		src = src * math.sqrt(self.n_feats)
		src = self.pos_encoder(src)
		memory = self.transformer_encoder(src)
		tgt = tgt.repeat(1, 1, 2)
		return tgt, memory

	def forward(self, src, tgt):
		# Phase 1 - Without anomaly scores
		c = torch.zeros_like(src).to(device=global_device)
		o1 = self.fcn1(self.transformer_decoder1(*self.encode(src, c, tgt)))
		o2 = self.fcn2(self.transformer_decoder2(*self.encode(src, c, tgt)))
		# Phase 2 - With anomaly scores
		c2 = (o1 - src) ** 2
		o2_hat = self.fcn3(self.transformer_decoder2(*self.encode(src, c2, tgt)))
		# print(f"o1:{o1.shape} o2_hat:{o2_hat.shape} o2:{o2.shape}")
		return o1, o2, o2_hat

def freeze_a_layer(layer: nn.Module):
    for param in layer.parameters():
        param.requires_grad = False
def unfreeze_a_layer(layer: nn.Module):
    for param in layer.parameters():
        param.requires_grad = True
class TrainTranAD:
	def __init__(self, feats, max_epoch) -> None:
		self.feats = feats
		self.model = TranAD(feats).to(device=global_device)
		self.max_epoch = max_epoch
		self.optimizer_G = torch.optim.AdamW(self.model.parameters() , lr=self.model.lr, weight_decay=1e-5)
		self.optimizer_D = torch.optim.AdamW(self.model.parameters() , lr=self.model.lr, weight_decay=1e-5)
		self.scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer_G, T_max=40, eta_min=1e-5)
		self.scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer_D, T_max=40, eta_min=1e-5)
		self.loss_func = nn.MSELoss(reduction = 'none')
		self.best_loss = 999999999999
	
	def freeze_layers(self, freeze_layer: str = None):
		if freeze_layer == 'freeze_encoder':
			freeze_a_layer(self.model.pos_encoder)
			freeze_a_layer(self.model.transformer_encoder)
			freeze_a_layer(self.model.fcn1)
		if freeze_layer == 'freeze_att':
			self.model.transformer_encoder.layers[0].self_attn.in_proj_weight.requires_grad = False
			self.model.transformer_decoder1.layers[0].self_attn.in_proj_weight.requires_grad = False
			self.model.transformer_decoder1.layers[0].multihead_attn.in_proj_weight.requires_grad = False
			self.model.transformer_decoder2.layers[0].self_attn.in_proj_weight.requires_grad = False
			self.model.transformer_decoder2.layers[0].multihead_attn.in_proj_weight.requires_grad = False
	def unfreeze_layers(self, freeze_layer: str = None):
		unfreeze_a_layer(self.model.pos_encoder)
		unfreeze_a_layer(self.model.transformer_encoder)
		unfreeze_a_layer(self.model.fcn1)
		self.model.transformer_encoder.layers[0].self_attn.in_proj_weight.requires_grad = True
		self.model.transformer_decoder1.layers[0].self_attn.in_proj_weight.requires_grad = True
		self.model.transformer_decoder1.layers[0].multihead_attn.in_proj_weight.requires_grad = True
		self.model.transformer_decoder2.layers[0].self_attn.in_proj_weight.requires_grad = True
		self.model.transformer_decoder2.layers[0].multihead_attn.in_proj_weight.requires_grad = True	
	def init_layers(self,):
		# self.model.encoder_layers.linear1.reset_parameters()
		# self.model.encoder_layers.linear2.reset_parameters()
		# self.model.decoder_layers1.linear1.reset_parameters()
		self.model.decoder_layers1.linear2.reset_parameters()
		# self.model.decoder_layers2.linear1.reset_parameters()
		self.model.decoder_layers2.linear2.reset_parameters()
		# self.model.fcn1[0].reset_parameters()
		# self.model.fcn1[2].reset_parameters()
		# self.model.fcn2[0].reset_parameters()
		# self.model.fcn2[2].reset_parameters()
		# self.model.fcn3[0].reset_parameters()
		self.model.fcn3[2].reset_parameters()		


	def compose_loss(self,k,o1,o2,o2_hat,elem):
		index_loss_weight_tensor = torch.tensor(index_loss_weight).to(global_device)
		mse_left_G = self.loss_func(o1, elem)
		mse_left_G = torch.mean(mse_left_G.mul(index_loss_weight_tensor)/torch.sum(index_loss_weight_tensor))

		mse_right_G = self.loss_func(o2_hat, elem)
		mse_right_G = torch.mean(mse_right_G.mul(index_loss_weight_tensor)/torch.sum(index_loss_weight_tensor))
		loss_G =k* mse_left_G + (1 - k) * mse_right_G

		mse_left_D = self.loss_func(o2, elem)
		mse_left_D = torch.mean(mse_left_D.mul(index_loss_weight_tensor)/torch.sum(index_loss_weight_tensor))
		mse_right_D = self.loss_func(o2_hat, elem)
		mse_right_D = torch.mean(mse_right_D.mul(index_loss_weight_tensor)/torch.sum(index_loss_weight_tensor))
		loss_D = k* mse_left_D - (1 - k) * mse_right_D

		return loss_G,loss_D
	def valid_log(self, valid_data, log_file, save_path, epoch):
		l1_list, l2_list, obj_list, recon_list = [], [], [], []
		with torch.no_grad():
			for d in valid_data:
				local_bs = d.shape[0] # batch_size
				window = d.permute(1, 0, 2).to(device=global_device) # time;batch;feature
				elem = window[-1, :, :].view(1, local_bs, self.feats).to(device=global_device) 
				o1, o2, o2_hat = self.model(window, elem) 
				epsilon = 0.95
				k = epsilon**epoch
				# k = 1/ epoch
				loss_G,loss_D = self.compose_loss(k,o1,o2,o2_hat,elem)
				# loss_G = k * self.loss_func(o1, elem) + (1 - k) * self.loss_func(o2_hat, elem)
				# loss_D = k * self.loss_func(o2, elem) - (1 - k) * self.loss_func(o2_hat, elem)
				l1_list.append(torch.mean(loss_G).item())
				l2_list.append(torch.mean(loss_D).item())
				obj_list.append(torch.mean(self.loss_func(o2_hat, elem)).item())
				recon_list.append(torch.mean(self.loss_func(o1, elem)).item())
			obj = np.mean(obj_list)
			recon = np.mean(recon_list)
			# if obj<self.best_loss:
			if recon<self.best_loss:
				# self.best_loss = obj
				self.best_loss = recon
				self.save(save_path)
				print(f"***", end="")
				print(f"***", end="", file=log_file)
			print(f'Epoch {epoch}, obj:{obj}, recon:{recon}')
			print(f'Epoch {epoch}, obj:{obj} recon:{recon}', file=log_file)


	def fit(self, train_values, save_dir: pathlib.Path, valid_portion):
		train_sliding_window = SlidingWindowDataLoader(
			SlidingWindowDataset(train_values, self.model.n_window, if_wrap=True),
			batch_size=self.model.batch,
			shuffle=True,
			drop_last=False
		)
		all_data = list(train_sliding_window)
		print(f"all_data:{len(all_data)} {all_data[0].shape}")
		if len(all_data) > 1:
			valid_index_list = [i for i in range(0, len(all_data), int(1/valid_portion))]
			train_index_list = [i for i in range(0, len(all_data)) if i not in valid_index_list]
			train_data = np.array(all_data)[train_index_list]
			valid_data = np.array(all_data)[valid_index_list]
		else:
			train_data = all_data
			valid_data = all_data

		# if (save_dir/"log.txt").exists():
		# 	os.remove(save_dir/"log.txt")
		log_file = open(save_dir/"log.txt", mode='a+')
		start_epoch=global_start_epoch
		self.valid_log(valid_data, log_file, save_dir, 0)
		for epoch in range(start_epoch, self.max_epoch+start_epoch):
			for d in train_data:
				local_bs = d.shape[0] # batch_size
				window = d.permute(1, 0, 2).to(device=global_device) # time;batch;feature
				elem = window[-1, :, :].view(1, local_bs, self.feats).to(device=global_device) 
				o1, o2, o2_hat = self.model(window, elem)
				epsilon = global_epsilon
				k = epsilon**epoch
				# k = 1/ epoch
				loss_G,loss_D = self.compose_loss(k,o1,o2,o2_hat,elem)
				# loss_G = k * self.loss_func(o1, elem) + (1 - k) * self.loss_func(o2_hat, elem)
				# loss_D = k * self.loss_func(o2, elem) - (1 - k) * self.loss_func(o2_hat, elem)
				self.optimizer_G.zero_grad()
				self.optimizer_D.zero_grad()
				torch.mean(loss_G).backward(retain_graph=True)
				torch.mean(loss_D).backward(retain_graph=False)
				self.optimizer_G.step()
				self.optimizer_D.step()

			# self.scheduler_G.step()
			# self.scheduler_D.step()
			if epoch % global_valid_epoch_freq == 0:
				self.valid_log(valid_data, log_file, save_dir, epoch)
        

	
	def predict(self, test_values):
		test_sliding_window = SlidingWindowDataLoader(
			SlidingWindowDataset(test_values, self.model.n_window),
			batch_size=self.model.batch,
			shuffle=False,
			drop_last=False
		)
		score_list_G = None
		score_list_GD = None
		o1_list = None
		o2_hat_list = None
		with torch.no_grad():
			for d in test_sliding_window:
				local_bs = d.shape[0] # batch_size
				window = d.permute(1, 0, 2).to(device=global_device)
				elem = window[-1, :, :].view(1, local_bs, self.feats).to(device=global_device)
				o1, o2, o2_hat = self.model(window, elem)
				score_G = ((o1-elem)**2).cpu().numpy()
				score_GD = ((o2_hat-elem)**2).cpu().numpy()
				if score_list_G is None:
					score_list_G = score_G
					score_list_GD = score_GD
					o1_list = o1.cpu().numpy()
					o2_hat_list = o2_hat.cpu().numpy()
				else:
					# print(f"o1_list:{o1_list.shape} o1:{o1.shape}")
					score_list_G = np.concatenate([score_list_G, score_G], axis=1)
					score_list_GD = np.concatenate([score_list_GD, score_GD], axis=1)
					o1_list = np.concatenate([o1_list, o1.cpu().numpy()], axis=1)
					o2_hat_list = np.concatenate([o2_hat_list, o2_hat.cpu().numpy()], axis=1)
		# print(f"score_list:{score_list.shape} o1_list:{o1_list.shape} o2_hat_list:{o2_hat_list.shape}")
		return np.squeeze(score_list_G, axis=0), np.squeeze(score_list_GD, axis=0), np.squeeze(o1_list, axis=0), np.squeeze(o2_hat_list, axis=0)
	
	def save(self, save_dir: pathlib.Path):
		torch.save(self.model.state_dict(), save_dir/"model.pth")
	def restore(self, save_dir: pathlib.Path):
		self.model.load_state_dict(torch.load(save_dir/"model.pth"))