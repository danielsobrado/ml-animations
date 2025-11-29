package diffusion

import (
	"math/rand"
)

// ResBlock is a residual block with time embedding.
type ResBlock struct {
	Conv1    *Conv2d
	Conv2    *Conv2d
	Norm1    *GroupNorm
	Norm2    *GroupNorm
	TimeMLP  *Linear
	SkipConv *Conv2d
	Channels int
}

// NewResBlock creates a new residual block.
func NewResBlock(inChannels, outChannels, timeEmbDim int, rng *rand.Rand) *ResBlock {
	var skipConv *Conv2d
	if inChannels != outChannels {
		skipConv = NewConv2d(inChannels, outChannels, 1, 1, 0, rng)
	}

	numGroups := 8
	if outChannels < 8 {
		numGroups = outChannels
	}

	return &ResBlock{
		Conv1:    NewConv2d(inChannels, outChannels, 3, 1, 1, rng),
		Conv2:    NewConv2d(outChannels, outChannels, 3, 1, 1, rng),
		Norm1:    NewGroupNorm(numGroups, outChannels),
		Norm2:    NewGroupNorm(numGroups, outChannels),
		TimeMLP:  NewLinear(timeEmbDim, outChannels, rng),
		SkipConv: skipConv,
		Channels: outChannels,
	}
}

// Forward computes the residual block output.
func (r *ResBlock) Forward(x, timeEmb *Tensor) *Tensor {
	batch := x.Shape[0]
	height := x.Shape[2]
	width := x.Shape[3]

	// First conv + norm + activation
	h := r.Conv1.Forward(x)
	h = r.Norm1.Forward(h)
	h = h.SiLU()

	// Add time embedding
	timeProj := r.TimeMLP.Forward(timeEmb)
	timeProj = timeProj.SiLU()

	// Broadcast time projection to spatial dimensions
	// timeProj shape: [batch, channels] -> need to add to [batch, channels, height, width]
	for b := 0; b < batch; b++ {
		for c := 0; c < r.Channels; c++ {
			timeVal := timeProj.At(b, c)
			for hi := 0; hi < height; hi++ {
				for wi := 0; wi < width; wi++ {
					val := getVal4D(h, b, c, hi, wi)
					setVal4D(h, val+timeVal, b, c, hi, wi)
				}
			}
		}
	}

	// Second conv + norm + activation
	h = r.Conv2.Forward(h)
	h = r.Norm2.Forward(h)
	h = h.SiLU()

	// Skip connection
	skip := x
	if r.SkipConv != nil {
		skip = r.SkipConv.Forward(x)
	}

	return h.Add(skip)
}

// ParameterCount returns total parameters.
func (r *ResBlock) ParameterCount() int {
	count := r.Conv1.ParameterCount() + r.Conv2.ParameterCount()
	count += r.Norm1.ParameterCount() + r.Norm2.ParameterCount()
	count += r.TimeMLP.ParameterCount()
	if r.SkipConv != nil {
		count += r.SkipConv.ParameterCount()
	}
	return count
}

// Downsample reduces spatial dimensions by 2.
type Downsample struct {
	Conv *Conv2d
}

// NewDownsample creates a downsample block.
func NewDownsample(channels int, rng *rand.Rand) *Downsample {
	return &Downsample{
		Conv: NewConv2d(channels, channels, 3, 2, 1, rng),
	}
}

// Forward downsamples the input.
func (d *Downsample) Forward(x *Tensor) *Tensor {
	return d.Conv.Forward(x)
}

// Upsample increases spatial dimensions by 2.
type Upsample struct {
	Conv *Conv2d
}

// NewUpsample creates an upsample block.
func NewUpsample(channels int, rng *rand.Rand) *Upsample {
	return &Upsample{
		Conv: NewConv2d(channels, channels, 3, 1, 1, rng),
	}
}

// Forward upsamples the input using nearest neighbor interpolation.
func (u *Upsample) Forward(x *Tensor) *Tensor {
	batch := x.Shape[0]
	channels := x.Shape[1]
	height := x.Shape[2]
	width := x.Shape[3]

	newHeight := height * 2
	newWidth := width * 2

	upsampled := Zeros([]int{batch, channels, newHeight, newWidth})

	for b := 0; b < batch; b++ {
		for c := 0; c < channels; c++ {
			for h := 0; h < height; h++ {
				for w := 0; w < width; w++ {
					val := getVal4D(x, b, c, h, w)
					// Copy to 2x2 block
					setVal4D(upsampled, val, b, c, h*2, w*2)
					setVal4D(upsampled, val, b, c, h*2+1, w*2)
					setVal4D(upsampled, val, b, c, h*2, w*2+1)
					setVal4D(upsampled, val, b, c, h*2+1, w*2+1)
				}
			}
		}
	}

	return u.Conv.Forward(upsampled)
}

// UNet is a simplified U-Net for noise prediction.
type UNet struct {
	// Initial projection
	InitConv *Conv2d

	// Time embedding
	TimeEmb1   *Linear
	TimeEmb2   *Linear
	TimeEmbDim int

	// Encoder
	DownBlocks  []*ResBlock
	Downsamples []*Downsample

	// Middle
	MidBlock1 *ResBlock
	MidBlock2 *ResBlock

	// Decoder
	UpBlocks  []*ResBlock
	Upsamples []*Upsample

	// Final projection
	FinalNorm *GroupNorm
	FinalConv *Conv2d

	rng *rand.Rand
}

// NewUNet creates a new U-Net.
func NewUNet(
	inChannels int,
	modelChannels int,
	channelMult []int,
	numResBlocks int,
	rng *rand.Rand,
) *UNet {
	timeEmbDim := modelChannels * 4

	unet := &UNet{
		TimeEmbDim: timeEmbDim,
		rng:        rng,
	}

	// Time embedding MLPs
	unet.TimeEmb1 = NewLinear(modelChannels, timeEmbDim, rng)
	unet.TimeEmb2 = NewLinear(timeEmbDim, timeEmbDim, rng)

	// Initial conv
	unet.InitConv = NewConv2d(inChannels, modelChannels, 3, 1, 1, rng)

	// Build encoder
	ch := modelChannels
	for level, mult := range channelMult {
		outCh := modelChannels * mult

		for i := 0; i < numResBlocks; i++ {
			unet.DownBlocks = append(unet.DownBlocks, NewResBlock(ch, outCh, timeEmbDim, rng))
			ch = outCh
		}

		if level < len(channelMult)-1 {
			unet.Downsamples = append(unet.Downsamples, NewDownsample(ch, rng))
		}
	}

	// Middle blocks
	unet.MidBlock1 = NewResBlock(ch, ch, timeEmbDim, rng)
	unet.MidBlock2 = NewResBlock(ch, ch, timeEmbDim, rng)

	// Build decoder (reverse of encoder)
	for level := len(channelMult) - 1; level >= 0; level-- {
		mult := channelMult[level]
		outCh := modelChannels * mult

		for i := 0; i < numResBlocks; i++ {
			// Double channels for skip connections
			skipCh := ch
			if i == 0 && level < len(channelMult)-1 {
				skipCh = ch * 2
			}
			_ = skipCh
			unet.UpBlocks = append(unet.UpBlocks, NewResBlock(ch, outCh, timeEmbDim, rng))
			ch = outCh
		}

		if level > 0 {
			unet.Upsamples = append(unet.Upsamples, NewUpsample(ch, rng))
		}
	}

	// Final layers
	numGroups := 8
	if ch < 8 {
		numGroups = ch
	}
	unet.FinalNorm = NewGroupNorm(numGroups, ch)
	unet.FinalConv = NewConv2d(ch, inChannels, 3, 1, 1, rng)

	return unet
}

// Forward performs the U-Net forward pass.
func (u *UNet) Forward(x *Tensor, timesteps []int) *Tensor {
	// Get timestep embedding
	timeEmb := GetTimestepEmbedding(timesteps, u.TimeEmbDim/4)
	timeEmb = u.TimeEmb1.Forward(timeEmb)
	timeEmb = timeEmb.SiLU()
	timeEmb = u.TimeEmb2.Forward(timeEmb)

	// Initial conv
	h := u.InitConv.Forward(x)

	// Store for skip connections
	skips := []*Tensor{h.Clone()}

	// Encoder
	blockIdx := 0
	downIdx := 0
	for level := 0; level < len(u.Downsamples)+1; level++ {
		numBlocks := 1 // Simplified
		for i := 0; i < numBlocks && blockIdx < len(u.DownBlocks); i++ {
			h = u.DownBlocks[blockIdx].Forward(h, timeEmb)
			skips = append(skips, h.Clone())
			blockIdx++
		}
		if downIdx < len(u.Downsamples) {
			h = u.Downsamples[downIdx].Forward(h)
			skips = append(skips, h.Clone())
			downIdx++
		}
	}

	// Middle
	h = u.MidBlock1.Forward(h, timeEmb)
	h = u.MidBlock2.Forward(h, timeEmb)

	// Decoder (simplified - not using full skip connections for clarity)
	upBlockIdx := 0
	upIdx := 0
	for level := 0; level < len(u.Upsamples)+1; level++ {
		numBlocks := 1
		for i := 0; i < numBlocks && upBlockIdx < len(u.UpBlocks); i++ {
			h = u.UpBlocks[upBlockIdx].Forward(h, timeEmb)
			upBlockIdx++
		}
		if upIdx < len(u.Upsamples) {
			h = u.Upsamples[upIdx].Forward(h)
			upIdx++
		}
	}

	// Final
	h = u.FinalNorm.Forward(h)
	h = h.SiLU()
	h = u.FinalConv.Forward(h)

	return h
}

// ParameterCount returns total parameters in the U-Net.
func (u *UNet) ParameterCount() int {
	count := u.InitConv.ParameterCount()
	count += u.TimeEmb1.ParameterCount() + u.TimeEmb2.ParameterCount()

	for _, b := range u.DownBlocks {
		count += b.ParameterCount()
	}
	for _, d := range u.Downsamples {
		count += d.Conv.ParameterCount()
	}

	count += u.MidBlock1.ParameterCount() + u.MidBlock2.ParameterCount()

	for _, b := range u.UpBlocks {
		count += b.ParameterCount()
	}
	for _, up := range u.Upsamples {
		count += up.Conv.ParameterCount()
	}

	count += u.FinalNorm.ParameterCount() + u.FinalConv.ParameterCount()

	return count
}
