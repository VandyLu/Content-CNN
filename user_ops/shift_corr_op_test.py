import tensorflow as tf

class ShiftCorrTest(tf.test.TestCase):
    def testShiftCorr(self):
        shift_corr_module = tf.load_op_library('./shift_corr.so')
        a = tf.reshape(tf.range(16,dtype=tf.float32),(2,2,2,2))
        b = tf.reshape(tf.range(16,dtype=tf.float32),(2,2,2,2))
        result = shift_corr_module.shift_corr(a,b,dispmax=3)
        result = tf.reshape(result,[-1])
        with self.test_session():
            #self.assertAllEqual(result.eval(),[5,0,0,0,0])
            print result.eval()

if __name__ == '__main__':
    tf.test.main()
