from com.path_helper import *
from lb_cloth_3d.beta_2_cloth_mesh.loader import *
from neural_network.ground_truth import *

'''
base = r'D:\Work\Cloth\testing_training_lp_tpose' + '\\'

test_dir = base + 'testing.txt'
train_dir = base + 'training.txt'
faces = base + 'triangles.txt'
betas = base + 'BetaTraining.txt'
'''

from env.shape_path import *


class AutoEncoderGroundTruth(GroundTruth):

    def __init__(self):

        self.triangles = txt_to_array_fixed_width(faces, int, 3) - 1
        self.zero_beta_index = -1

        self.train = {
            'betas': txt_to_array_fixed_shape(betas, float, (17, 4)),
            'verts': []
        }

        for i in range(17):
            cloth = join(train_dir, 'cloth_%s.txt' % str3(i))
            verts = txt_to_array_fixed_width(cloth, float, 3)
            self.train['verts'].append(verts)

            if (self.train['betas'][i] == [0, 0, 0, 0]).all():
                self.zero_beta_index = i

        self.template = np.copy(self.train['verts'][self.zero_beta_index])

        for i in range(17):
            self.train['verts'][i] -= self.template

        self.test = {
            'betas': np.array([]),
            'verts': []
        }

        bs = []
        for i in range(12):
            cloth = join(test_dir, 'sim_cloth_%s.txt' % str3(i))
            verts = txt_to_array_fixed_width(cloth, float, 3)
            beta = join(test_dir, 'beta_%s.txt' % str3(i))
            bs.append(txt_to_array_fixed_shape(beta, float, (4,), '\t'))
            self.test['verts'].append(verts - self.template)
        self.test['betas'] = np.array(bs).reshape(12, 4)

        self.batch_manager = BatchManager(17 + 11, 17)

    def get_batch(self, size):
        ids = self.batch_manager.get_batch_samples(size)

        def i_2_s(i):
            return [self.train['betas'][i], np.array(self.train['verts'][i]).reshape(-1)]

        batch = ids_2_batch(i_2_s, ids)
        return batch

    def get_test(self):
        ids = range(12)

        def i_2_s(i):
            return [self.test['betas'][i], np.array(self.test['verts'][i]).reshape(-1)]

        batch = ids_2_batch(i_2_s, ids)
        return batch



def set_smooth_times(i):
    global smooth_times
    smooth_times = i


if __name__ == '__main__':
    """
    """
    aegt =  AutoEncoderGroundTruth()
    print(aegt.zero_beta_index)
    print(aegt.get_batch(10))
    Mesh().from_vertices(aegt.train['verts'][0] * (np.random.rand(*aegt.template.shape) - 0.5) * 0.1 + aegt.template, aegt.triangles).save('tst.obj')
