from pymediainfo import MediaInfo
import subprocess

from torchvision import transforms as trn
from torchvision.transforms.functional import resize
from PIL import Image

from nets.models import *
from torch.utils.data import DataLoader, Dataset


class ListDataset(Dataset):
    def __init__(self, l, transform=None):
        self.l = l
        default_transform = trn.Compose([
            trn.Resize((224, 224)),
            trn.ToTensor(),
            trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transform = transform or default_transform

    def __getitem__(self, idx):
        return self.transform(self.l[idx])

    def __len__(self):
        return len(self.l)

    def __repr__(self):
        fmt_str = f'{self.__class__.__name__}\n'
        fmt_str += f'\tNumber of images : {self.__len__()}\n'
        trn_str = self.transform.__repr__().replace('\n', '\n\t')
        fmt_str += f"\tTransform : \n\t{trn_str}"

        return fmt_str


def parse_metadata(path):
    media_info = MediaInfo.parse(path)
    meta = {'file_path': path}
    for track in media_info.tracks:
        if track.track_type == 'General':
            meta['file_name'] = track.file_name + '.' + track.file_extension
            meta['file_extension'] = track.file_extension
            meta['format'] = track.format
            meta['duration'] = float(track.duration)
            meta['frame_count'] = int(track.frame_count)
            meta['frame_rate'] = float(track.frame_rate)
        elif track.track_type == 'Video':
            meta['width'] = int(track.width)
            meta['height'] = int(track.height)
            meta['rotation'] = float(track.rotation or 0.)
            meta['codec'] = track.codec
    return meta


def decode_frames(video, meta, decode_rate, size):
    frames = []
    w, h = (meta['width'], meta['height']) if meta['rotation'] not in [90, 270] else (meta['height'], meta['width'])
    command = ['ffmpeg',
               '-hide_banner', '-loglevel', 'panic',
               '-vsync', '2',
               '-i', video,
               '-pix_fmt', 'bgr24',  # color space
               '-r', str(decode_rate),
               '-q:v', '0',
               '-vcodec', 'rawvideo',  # origin video
               '-f', 'image2pipe',  # output format : image to pipe
               'pipe:1']
    pipe = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=w * h * 3)
    while True:
        raw_image = pipe.stdout.read(w * h * 3)
        pipe.stdout.flush()
        try:
            image = Image.frombuffer('RGB', (w, h), raw_image, "raw", 'BGR', 0, 1)
        except ValueError as e:
            break

        if size:
            image = resize(image, size)
        frames.append(image)
    return frames


@torch.no_grad()
def extract_frame_fingerprint(model, loader):
    model.eval()
    frame_fingerprints = []

    for im in loader:
        feature = model(im)
        frame_fingerprints.append(feature.cpu())
    frame_fingerprints = torch.cat(frame_fingerprints)
    return frame_fingerprints


if __name__ == '__main__':
    video = '/nfs_shared/MLVD/VCDB/videos/00274a923e13506819bd273c694d10cfa07ce1ec.flv'
    decode_rate = 10
    decode_size = 256
    group_count = 4
    cnn_model = MobileNet_AVG().cuda()
    cnn_model = nn.DataParallel(cnn_model)
    aggr_model = Segment_Maxpooling()

    # parse video metadata
    meta = parse_metadata(video)
    print(meta)

    # decode all frames
    frames = decode_frames(video, meta, decode_rate, decode_size)
    print(len(frames))

    # extract frame fingerprint
    transform = trn.Compose([
        trn.Resize((224, 224)),
        trn.ToTensor(),
        trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    cnn_loader = DataLoader(ListDataset(frames, transform=transform), batch_size=64, shuffle=False, num_workers=4)
    frame_fingerprints = extract_frame_fingerprint(cnn_model, cnn_loader)
    print(frame_fingerprints.shape)

    # grouping fingerprints for each segment => If frame_fingerprints cannot be divided by group_count, the last is copied.
    k = group_count - frame_fingerprints.shape[0] % group_count
    if k != group_count:
        frame_fingerprints = torch.cat([frame_fingerprints, frame_fingerprints[-1:, ].repeat((k, 1))])
    frame_fingerprints = frame_fingerprints.reshape(-1, group_count, frame_fingerprints.shape[-1])
    print(frame_fingerprints.shape)

    # extract segment_fingerprint
    frame_fingerprints = frame_fingerprints.permute(0, 2, 1)
    print(frame_fingerprints.shape)
    segment_fingerprints = aggr_model(frame_fingerprints)
    print(segment_fingerprints.shape)
