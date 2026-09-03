import torch
import numpy as np

def extract_ampl_phase(fft_im):
    # fft_im: size should be bx3xhxwx2
    fft_amp = fft_im[:,:,:,:,0]**2 + fft_im[:,:,:,:,1]**2
    fft_amp = torch.sqrt(fft_amp)
    fft_pha = torch.atan2( fft_im[:,:,:,:,1], fft_im[:,:,:,:,0] )
    return fft_amp, fft_pha

def low_freq_mutate( amp_src, amp_trg, L=0.1 ):
    _, _, h, w = amp_src.size()
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)     # get b
    amp_src[:,:,0:b,0:b]     = amp_trg[:,:,0:b,0:b]      # top left
    amp_src[:,:,0:b,w-b:w]   = amp_trg[:,:,0:b,w-b:w]    # top right
    amp_src[:,:,h-b:h,0:b]   = amp_trg[:,:,h-b:h,0:b]    # bottom left
    amp_src[:,:,h-b:h,w-b:w] = amp_trg[:,:,h-b:h,w-b:w]  # bottom right
    return amp_src

def low_freq_mutate_np( amp_src, amp_trg, L=0.1 ):
    a_src = np.fft.fftshift( amp_src, axes=(-2, -1) )
    a_trg = np.fft.fftshift( amp_trg, axes=(-2, -1) )

    _, h, w = a_src.shape
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)

    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    a_src[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2]
    a_src = np.fft.ifftshift( a_src, axes=(-2, -1) )
    return a_src

def FDA_source_to_target(src_img, trg_img, L=0.1):
    # exchange magnitude
    # input: src_img, trg_img
    img_min = torch.min(src_img)
    img_max = torch.max(src_img)

    # 1. Get FFT of both source and target (Modern PyTorch 2.1.0)
    # Removed .clone() to prevent massive memory spikes
    # view_as_real maintains the [B, C, H, W, 2] shape expected by downstream functions
    fft_src = torch.view_as_real(torch.fft.fft2(src_img))
    fft_trg = torch.view_as_real(torch.fft.fft2(trg_img))

    # 2. Extract amplitude and phase of both FFTs
    amp_src, pha_src = extract_ampl_phase(fft_src)
    amp_trg, pha_trg = extract_ampl_phase(fft_trg)

    # 3. Replace the low frequency amplitude part of source with that from target
    amp_src_ = low_freq_mutate(amp_src, amp_trg, L=L)

    # 4. Recompose FFT of source
    # Added `device=src_img.device` to ensure it doesn't default to CPU if inputs are on GPU
    fft_src_ = torch.zeros(fft_src.size(), dtype=torch.float, device=src_img.device)
    
    # Using `...` is a cleaner way to apply this to the last dimension
    fft_src_[..., 0] = torch.cos(pha_src) * amp_src_
    fft_src_[..., 1] = torch.sin(pha_src) * amp_src_

    # 5. Get the recomposed image: source content, target style
    _, _, imgH, imgW = src_img.size()
    
    # --- THE FIX FOR THE 5-CHANNEL SHAPE BUG ---
    # Convert the [..., 2] float tensor back to a true complex tensor
    complex_fft_src = torch.view_as_complex(fft_src_)
    
    # Perform the inverse FFT and take the real part
    src_in_trg = torch.fft.ifft2(complex_fft_src, s=(imgH, imgW)).real
    
    # Clamp the output to the original image's value range to prevent outliers
    src_in_trg = torch.clamp(src_in_trg, img_min, img_max)

    return src_in_trg

def FDA_source_to_target_np( src_img, trg_img, L=0.1 ):
    # exchange magnitude
    # input: src_img, trg_img

    src_img_np = src_img #.cpu().numpy()
    trg_img_np = trg_img #.cpu().numpy()

    # get fft of both source and target
    fft_src_np = np.fft.fft2( src_img_np, axes=(-2, -1) )
    fft_trg_np = np.fft.fft2( trg_img_np, axes=(-2, -1) )

    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)
    amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

    # mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_np( amp_src, amp_trg, L=L )

    # mutated fft of source
    fft_src_ = amp_src_ * np.exp( 1j * pha_src )

    # get the mutated image
    src_in_trg = np.fft.ifft2( fft_src_, axes=(-2, -1) )
    src_in_trg = np.real(src_in_trg)

    return src_in_trg

