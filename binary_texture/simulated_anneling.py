import sys
import random as rn
from math import exp

from PIL import Image
import numpy as np

import pygame

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# TODO: use rect everywhere
# TODO: only recompute patterns from modified rect 
class Rect:
    def __init__(self, x, y, w, h):
        self.x1 = x
        self.y1 = y
        self.w = w
        self.h = h
        self.x2 = x + w
        self.y2 = y + h

    def get_center(self):
        return ((int)((self.x1 + self.x2)/2), (int)((self.y1 + self.y2)/2))

    def intersect(self, other_rect):
        return (self.x1 <= other_rect.x2 and self.x2 >= other_rect.x1 and
                self.y1 <= other_rect.y2 and self.y2 >= other_rect.y1)

class Patch:
    def __init__(self, contents, top_left):
        self.contents = contents
        # TODO: add rect


class BinaryTextureSimulatedAnneling:
    def __init__(self, path=None, initial_temp=sys.maxsize,
                 min_block_size=1, max_block_size=2, dst_func='l1'):

        self.base_texture = None
        self.img_size = None
        self.generated_texture_last_iter = None
        self.generated_texture = None

        self.path = path
        if path:
            self.import_image(path)
        
        self.base_image_patches = {}

        self.initial_temp = initial_temp
        self.current_temp = initial_temp
        self.dst_func = dst_func

        self.min_block_size = min_block_size
        self.max_block_size = max_block_size

    def import_image(self, path):
        self.base_texture = Image.open(path).convert('L')
        self.img_size = list(self.base_texture.size)
        self.img_size.reverse()
        self.base_texture = np.asarray(self.base_texture).copy()
        self.base_texture[self.base_texture > 1] = 1

    def create_random_rect(self):
        block_size = rn.randint(2, min(self.img_size) - 1)

        y_ori = rn.randint(0, self.img_size[0] - block_size - 1)
        x_ori = rn.randint(0, self.img_size[1] - block_size - 1)

        rect = Rect(x_ori, y_ori, block_size, block_size)
        return rect

    def create_base_generated_texture(self, desired_size):
        self.generated_texture = np.random.randint(2, size=desired_size)
        self.generated_texture_last_iter = self.generated_texture.copy()

    def modify_texture(self):
        self.generated_texture_last_iter = self.generated_texture.copy()

        #TODO: use create_random_rect()
        block_size = rn.randint(2, min(self.img_size) - 1)

        y_ori = rn.randint(0, self.img_size[0] - block_size - 1)
        x_ori = rn.randint(0, self.img_size[1] - block_size - 1)

        y_dest = rn.randint(0, self.generated_texture.shape[0] - block_size - 1)
        x_dest = rn.randint(0, self.generated_texture.shape[1] - block_size - 1)

        self.generated_texture[y_dest : y_dest + block_size - 1,
                               x_dest : x_dest + block_size - 1] = self.base_texture[y_ori : y_ori + block_size - 1,
                                                                                     x_ori : x_ori + block_size - 1]

    def get_base_image_patches(self, block_size):
        """
            Get all the patches for the desired block size
            in the base texture (only do it once).
        """
        base_image_patches = []

        for y in range(0, self.img_size[0] - block_size):
            for x in range(0, self.img_size[1] - block_size):
                sub_matrix = self.base_texture[y : y + block_size, x : x + block_size]

                found = False

                for patch in base_image_patches:
                    if (sub_matrix == patch).all():
                        found = True
                        break
                
                if not found:
                    base_image_patches.append(sub_matrix)

        self.base_image_patches[block_size] = base_image_patches

    def get_all_patches(self, block_size):
        all_patches = self.base_image_patches[block_size].copy()

        for y in range(0, self.generated_texture.shape[0] - block_size):
            for x in range(0, self.generated_texture.shape[1] - block_size):
                sub_matrix = self.generated_texture[y : y + block_size, x : x + block_size]

                found = False

                for patch in all_patches:
                    if (sub_matrix == patch).all():
                        found = True
                        break
                
                if not found:
                    all_patches.append(sub_matrix)

        return all_patches

    def find_matrix_index_in_list(self, matrix, lst):
        for i, lst_mat in enumerate(lst):
            if np.array_equal(lst_mat, matrix):
                return i

        return -1


    def get_patches_occurences(self, block_size):
        all_patches = self.get_all_patches(block_size)

        original_texture_patches_occurences = {k: 0 for k, _ in enumerate(all_patches)}
        generated_texture_patches_occurences = {k: 0 for k, _ in enumerate(all_patches)}

        for y in range(0, self.img_size[0] - block_size):
            for x in range(0, self.img_size[1] - block_size):
                sub_matrix = self.base_texture[y : y + block_size, x : x + block_size]
                idx = self.find_matrix_index_in_list(sub_matrix, all_patches)
                original_texture_patches_occurences[idx] += 1

        summ = sum([v for v in original_texture_patches_occurences.values()])

        for k in original_texture_patches_occurences.keys():
            original_texture_patches_occurences[k] /= summ
        

        for y in range(0, self.generated_texture.shape[0] - block_size):
            for x in range(0, self.generated_texture.shape[1] - block_size):
                sub_matrix = self.generated_texture[y : y + block_size, x : x + block_size]
                idx = self.find_matrix_index_in_list(sub_matrix, all_patches)
                generated_texture_patches_occurences[idx] += 1

        summ = sum([v for v in generated_texture_patches_occurences.values()])

        for k in generated_texture_patches_occurences.keys():
            generated_texture_patches_occurences[k] /= summ
                
        return zip(list(original_texture_patches_occurences.values()), 
                   list(generated_texture_patches_occurences.values()))

    def one_pass_energy(self, block_size):
        patches_occurences = self.get_patches_occurences(block_size)

        if self.dst_func == 'l1':
            return sum([abs(v[0] - v[1]) for v in patches_occurences])

        elif self.dst_func == 'l2':
            return sum([pow(v[0] - v[1], 2) for v in patches_occurences])



    def compute_energy(self):
        energy = 0
        for bs in range(self.min_block_size, self.max_block_size + 1):
            energy += self.one_pass_energy(bs)
            
        return energy

    def init(self):
        for i in range(self.min_block_size, self.max_block_size + 1):
            self.get_base_image_patches(i)

    def main_loop(self):
        energy_values = []
        delta_energy_values = []
        temperatures_values = []

        energy_threshold = 0.001
        temp_threshold = 0.0000000000000001

        pygame.init()
        screen = pygame.display.set_mode((self.generated_texture.shape[0] * 10, self.generated_texture.shape[1] * 10))
        pygame.display.set_caption("Serious Work - not games")
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            before_energy = self.compute_energy()
            self.modify_texture()

            # Clear screen to white before drawing 
            screen.fill((255, 255, 255))
            new_array = self.generated_texture * 255
            surface = pygame.surfarray.make_surface(new_array)
            surface = pygame.transform.scale(surface, (self.generated_texture.shape[0] * 10, self.generated_texture.shape[1] * 10))
            screen.blit(surface, (0, 0))
            pygame.display.flip()

            after_energy = self.compute_energy()

            try:
                probability = exp(-(after_energy - before_energy) / self.current_temp)
            except OverflowError:
                probability = -1

            print(f'E(b) = {before_energy:.5f}, E(a) = {after_energy:.5f}, d(E) = {after_energy - before_energy:.5f}', end='')
            

            if after_energy - before_energy > 0:
                # Refusing changes
                print(f', probability: {probability:.5f}')
                if exp(-(after_energy - before_energy) / self.current_temp) < rn.random():
                    self.generated_texture = self.generated_texture_last_iter.copy()
                else:
                    energy_values.append(after_energy)
                    if not delta_energy_values:
                        delta_energy_values.append(after_energy - before_energy)
                    else:
                        delta_energy_values.append(delta_energy_values[-1] + (after_energy - before_energy))

            
            else:
                print(f', probability: accept')
                energy_values.append(after_energy)
                if not delta_energy_values:
                    delta_energy_values.append(after_energy - before_energy)
                else:
                    delta_energy_values.append(delta_energy_values[-1] + (after_energy - before_energy))

            self.current_temp = self.current_temp * 0.99
            print(f'Current temp: {self.current_temp}')
            
            temperatures_values.append(self.current_temp)

            if after_energy < energy_threshold:
                print(f'Energy reached threshold of {energy_threshold}')
                done = True

            if self.current_temp < temp_threshold:
                print(f'Temperature reached threshold of {temp_threshold}')
                done = True

        im = Image.fromarray(np.uint8(np.transpose(self.generated_texture * 255)))
        im.save("final_texture.png")

        df = pd.DataFrame({'iterations': [i for i in range(len(energy_values))],
                           'energy': energy_values,
                           'cumulative delta energy': delta_energy_values})

        
        ax = df.plot(x="iterations", y="energy", legend=False)
        ax2 = ax.twinx()
        var = (max(delta_energy_values) - min(delta_energy_values)) * 5
        ax2.set_ylim(np.mean(delta_energy_values) - var, np.mean(delta_energy_values) + var)
        df.plot(x="iterations", y="cumulative delta energy", ax=ax2, legend=False, color="r")
        ax.figure.legend()
        plt.show()
        fig = ax2.get_figure()
        fig.savefig('energy_graph.png')

        

def main():
    bt = BinaryTextureSimulatedAnneling(r'../test3.png', min_block_size=1, max_block_size=4,
                                        dst_func='l2')
    bt.create_base_generated_texture([32,32])
    bt.init()
    bt.main_loop()

if __name__ == '__main__':
    main()