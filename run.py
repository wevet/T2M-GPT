import clip
import torch
import numpy as np
import warnings
import datetime
 
import models.vqvae as vqvae
import models.t2m_trans as trans
import options.option_transformer as option_trans
from utils.motion_process import recover_from_ric
import visualization.plot_3d_global as plot_3d
 

clip_text = ["a person jumps forward 3m, stops for 2 seconds, turns left 45 degrees, and starts running."]
 
args = option_trans.get_args_parser()
args.dataname = 't2m'
args.resume_pth = 'pretrained/VQVAE/net_last.pth'
args.resume_trans = 'pretrained/VQTransformer_corruption05/net_best_fid.pth'
args.down_t = 2
args.depth = 3
args.block_size = 51
 
warnings.filterwarnings('ignore')
 
print('loading clip model \"ViT-B/32\"')
clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'), jit=True, download_root='./')
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

net = vqvae.HumanVQVAE(args, args.nb_code, args.code_dim, args.output_emb_width, args.down_t, args.stride_t, args.width,
                       args.depth, args.dilation_growth_rate)

trans_encoder = trans.Text2Motion_Transformer(num_vq=args.nb_code, embed_dim=1024, clip_dim=args.clip_dim,
                                              block_size=args.block_size, num_layers=9, n_head=16, drop_out_rate=args.drop_out_rate, fc_rate=args.ff_rate)
 
 
print('loading checkpoint from {}'.format(args.resume_pth))
ckpt = torch.load(args.resume_pth, map_location='cpu')
net.load_state_dict(ckpt['net'], strict=True)
net.eval()
net.cuda()
 
print('loading transformer checkpoint from {}'.format(args.resume_trans))
ckpt = torch.load(args.resume_trans, map_location='cpu')
trans_encoder.load_state_dict(ckpt['trans'], strict=True)
trans_encoder.eval()
trans_encoder.cuda()
 
mean = torch.from_numpy(np.load('./checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy')).cuda()
std = torch.from_numpy(np.load('./checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy')).cuda()
 
text = clip.tokenize(clip_text, truncate=True).cuda()
feat_clip_text = clip_model.encode_text(text).float()
index_motion = trans_encoder.sample(feat_clip_text[0:1], False)
pred_pose = net.forward_decoder(index_motion)
 
pred_xyz = recover_from_ric((pred_pose*std+mean).float(), 22)
xyz = pred_xyz.reshape(1, -1, 22, 3)
xyz_np = xyz.detach().cpu().numpy()

# 現在の日時を取得してフォーマット
timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")
filename = f"motion_{timestamp}.npy"
# npyファイルを保存
np.save(filename, xyz_np)
print(f"Motion data saved as {filename}")

# GIFの保存（ファイル名も変更する場合）
gif_filename = f"motion_{timestamp}.gif"
pose_vis = plot_3d.draw_to_batch(xyz_np, clip_text, [gif_filename])
print(f"GIF saved as {gif_filename}")
