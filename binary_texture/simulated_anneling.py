import sys
import random as rn
from math import exp

from PIL import Image
import numpy as np

import pygame

import pandas as pd
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
        return ((int)((self.x1 + self.x2) / 2), (int)((self.y1 + self.y2) / 2))

    def intersect(self, other_rect):
        return (self.x1 <= other_rect.x2 and self.x2 >= other_rect.x1 and
                self.y1 <= other_rect.y2 and self.y2 >= other_rect.y1)

class Patch:
    def __init__(self, contents, top_left):
        self.contents = contents
        self.dimensions = self.rect_from_top_left(top_left, contents.shape)

    def rect_from_top_left(self, top_left, content_size):
        x = top_left[0]
        y = top_left[1]
        w = content_size[1]
        h = content_size[0]
        rect = Rect(x, y, w, h)
        return rect


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

    def create_random_rect(self, block_size):
        y_ori = rn.randint(0, self.img_size[0] - block_size - 1)
        x_ori = rn.randint(0, self.img_size[1] - block_size - 1)

        rect = Rect(x_ori, y_ori, block_size, block_size)
        return rect

    def create_base_generated_texture(self, desired_size):
        self.generated_texture = np.random.randint(2, size=desired_size)
        self.generated_texture_last_iter = self.generated_texture.copy()

    def modify_texture(self):
        # TODO: if selected zone bigger than size, trim it
        self.generated_texture_last_iter = self.generated_texture.copy()

        block_size = rn.randint(2, min(self.img_size) - 1)
        input_zone = self.create_random_rect(block_size)

        x_origin_range = slice(input_zone.x1, input_zone.x1 + input_zone.w - 1)
        y_origin_range = slice(input_zone.y1, input_zone.y1 + input_zone.h - 1)

        y_dest = rn.randint(0, self.generated_texture.shape[0] - block_size - 1)
        x_dest = rn.randint(0, self.generated_texture.shape[1] - block_size - 1)

        y_dest_range = slice(y_dest, y_dest + block_size - 1)
        x_dest_range = slice(x_dest, x_dest + block_size - 1)

        self.generated_texture[y_dest_range,
                               x_dest_range] = self.base_texture[y_origin_range,
                                                                 x_origin_range]

        dest_rect = Rect(x_dest, y_dest,
                         x_dest + block_size - 1,
                         y_dest + block_size - 1)

        return dest_rect

    def get_base_image_patches(self, block_size):
        """
            Get all the patches for the desired block size
            in the base texture (only do it once).
        """
        base_image_patches = []

        for y in range(0, self.img_size[0] - block_size):
            for x in range(0, self.img_size[1] - block_size):
                sub_matrix = self.base_texture[y: y + block_size, x: x + block_size]

                found = False

                for patch in base_image_patches:
                    if (sub_matrix == patch.contents).all():
                        found = True
                        break

                if not found:
                    base_image_patches.append(Patch(sub_matrix, (x, y)))

        self.base_image_patches[block_size] = base_image_patches

    def get_all_patches(self, block_size, dest_rect):
        all_patches = self.base_image_patches[block_size].copy()

        # if dest_rect:
        #     # TODO:
        #     # - delete all patches which intersect with dest_rect
        #     # - re-create and recompute blocks that intersect with dest_rect
        #     pass



        for y in range(0, self.generated_texture.shape[0] - block_size):
            for x in range(0, self.generated_texture.shape[1] - block_size):
                sub_matrix = self.generated_texture[y: y + block_size, x: x + block_size]

                found = False

                for patch in all_patches:
                    if (sub_matrix == patch.contents).all():
                        found = True
                        break

                if not found:
                    all_patches.append(Patch(sub_matrix, (x, y)))

        return all_patches

    def find_matrix_index_in_list(self, matrix, lst):
        for i, lst_mat in enumerate(lst):
            if np.array_equal(lst_mat.contents, matrix):
                return i

        return -1


    def get_patches_occurences(self, block_size, dest_rect):
        all_patches = self.get_all_patches(block_size, dest_rect)

        original_texture_patches_occurences = {k: 0 for k, _ in enumerate(all_patches)}
        generated_texture_patches_occurences = {k: 0 for k, _ in enumerate(all_patches)}

        for y in range(0, self.img_size[0] - block_size):
            for x in range(0, self.img_size[1] - block_size):
                x_range = slice(x, x + block_size)
                y_range = slice(y, y + block_size)


                sub_matrix = self.base_texture[y_range, x_range]
                idx = self.find_matrix_index_in_list(sub_matrix, all_patches)
                original_texture_patches_occurences[idx] += 1

        summ = sum([v for v in original_texture_patches_occurences.values()])

        for k in original_texture_patches_occurences.keys():
            original_texture_patches_occurences[k] /= summ


        for y in range(0, self.generated_texture.shape[0] - block_size):
            for x in range(0, self.generated_texture.shape[1] - block_size):
                x_range = slice(x, x + block_size)
                y_range = slice(y, y + block_size)

                sub_matrix = self.generated_texture[y_range, x_range]
                idx = self.find_matrix_index_in_list(sub_matrix, all_patches)
                generated_texture_patches_occurences[idx] += 1

        summ = sum([v for v in generated_texture_patches_occurences.values()])

        for k in generated_texture_patches_occurences.keys():
            generated_texture_patches_occurences[k] /= summ

        return zip(list(original_texture_patches_occurences.values()),
                   list(generated_texture_patches_occurences.values()))

    def one_pass_energy(self, block_size, dest_rect):
        patches_occurences = self.get_patches_occurences(block_size, dest_rect)

        if self.dst_func == 'l1':
            return sum([abs(v[0] - v[1]) for v in patches_occurences])

        elif self.dst_func == 'l2':
            return sum([pow(v[0] - v[1], 2) for v in patches_occurences])



    def compute_energy(self, dest_rect=None):
        energy = 0
        for bs in range(self.min_block_size, self.max_block_size + 1):
            energy += self.one_pass_energy(bs, dest_rect)

        return energy

    def init(self):
        for i in range(self.min_block_size, self.max_block_size + 1):
            self.get_base_image_patches(i)

    def main_loop(self):
        energy_values = []
        delta_energy_values = []
        temperatures_values = []
        probability_values = []

        energy_threshold = 1e-3
        temp_threshold = 1e-16
        temp_multiplier = 0.999

        pygame.init()
        screen = pygame.display.set_mode((self.generated_texture.shape[0] * 10, self.generated_texture.shape[1] * 10))
        pygame.display.set_caption("Serious Work - not games")
        done = False

        after_energy = sys.maxsize

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            # Trick so that we don't compute energy two times
            before_energy = after_energy
            dest_rect = self.modify_texture()

            # Clear screen to white before drawing
            screen.fill((255, 255, 255))
            new_array = self.generated_texture * 255
            surface = pygame.surfarray.make_surface(new_array)
            surface = pygame.transform.scale(surface, (self.generated_texture.shape[0] * 10, self.generated_texture.shape[1] * 10))
            screen.blit(surface, (0, 0))
            pygame.display.flip()

            after_energy = self.compute_energy(dest_rect)

            try:
                probability = exp(-(after_energy - before_energy) / self.current_temp)
            except OverflowError:
                probability = -1

            probability_values.append(probability)
            energy_values.append(after_energy)

            if not delta_energy_values:
                delta_energy_values.append(after_energy - before_energy)
            else:
                delta_energy_values.append(delta_energy_values[-1] + (after_energy - before_energy))

            print(f'E(b) = {before_energy:.5f}, E(a) = {after_energy:.5f}, d(E) = {after_energy - before_energy:.5f}', end='')

            if after_energy - before_energy > 0:
                # Refusing changes
                print(f', probability: {probability:.5f}')
                if probability < rn.random():
                    self.generated_texture = self.generated_texture_last_iter.copy()
                # else:
                #     temp_multiplier *= 0.99

            else:
                print(f', probability: accept')


            print(f'Current temp: {self.current_temp} temp mult: {temp_multiplier:.2f}')

            self.current_temp *= temp_multiplier
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
                           'cumulative delta energy': delta_energy_values,
                           'probability of acceptation': probability_values})


        ax = df.plot(x="iterations", y="energy", legend=False)
        ax2 = ax.twinx()
        #var = (max(delta_energy_values) - min(delta_energy_values)) * 5
        #ax2.set_ylim(-1, 2)
        df.plot(x="iterations", y="cumulative delta energy", ax=ax2, legend=False, color="r")
        ax.figure.legend()
        plt.show()
        fig = ax2.get_figure()
        fig.savefig('energy_graph.png')



def main():
    bt = BinaryTextureSimulatedAnneling(r'../test3.png', min_block_size=1, max_block_size=4,
                                        dst_func='l2')
    bt.create_base_generated_texture([32, 16])
    bt.init()
    bt.main_loop()


if __name__ == '__main__':
    main()
