def match_passage_with_question(passage_reps, question_reps, passage_mask, question_mask, passage_lengths, question_lengths,
                                context_lstm_dim, scope=None,
                                with_full_match=True, with_maxpool_match=True, with_attentive_match=True, with_max_attentive_match=True,
                                is_training=True, options=None, dropout_rate=0, forward=True):
    passage_reps = tf.multiply(passage_reps, tf.expand_dims(passage_mask,-1))
    question_reps = tf.multiply(question_reps, tf.expand_dims(question_mask,-1))
    all_question_aware_representatins = []
    dim = 0
    with tf.variable_scope(scope or "match_passage_with_question"):
        relevancy_matrix = cal_relevancy_matrix(question_reps, passage_reps)
        relevancy_matrix = mask_relevancy_matrix(relevancy_matrix, question_mask, passage_mask)
        # relevancy_matrix = layer_utils.calcuate_attention(passage_reps, question_reps, context_lstm_dim, context_lstm_dim,
        #             scope_name="fw_attention", att_type=options.att_type, att_dim=options.att_dim,
        #             remove_diagnoal=False, mask1=passage_mask, mask2=question_mask, is_training=is_training, dropout_rate=dropout_rate)

        all_question_aware_representatins.append(tf.reduce_max(relevancy_matrix, axis=2,keep_dims=True))
        all_question_aware_representatins.append(tf.reduce_mean(relevancy_matrix, axis=2,keep_dims=True))
        dim += 2
        if with_full_match:
            if forward:
                question_full_rep = layer_utils.collect_final_step_of_lstm(question_reps, question_lengths - 1)
            else:
                question_full_rep = question_reps[:,0,:]

            passage_len = tf.shape(passage_reps)[1]
            question_full_rep = tf.expand_dims(question_full_rep, axis=1)
            question_full_rep = tf.tile(question_full_rep, [1, passage_len, 1])  # [batch_size, pasasge_len, feature_dim]

            (attentive_rep, match_dim) = multi_perspective_match(context_lstm_dim,
                                passage_reps, question_full_rep, is_training=is_training, dropout_rate=options.dropout_rate,
                                options=options, scope_name='mp-match-full-match')
            all_question_aware_representatins.append(attentive_rep)
            dim += match_dim

        if with_maxpool_match:
            maxpooling_decomp_params = tf.get_variable("maxpooling_matching_decomp",
                                                          shape=[options.cosine_MP_dim, context_lstm_dim], dtype=tf.float32)
            maxpooling_rep = cal_maxpooling_matching(passage_reps, question_reps, maxpooling_decomp_params)
            all_question_aware_representatins.append(maxpooling_rep)
            dim += 2*options.cosine_MP_dim

        if with_attentive_match:
            atten_scores = layer_utils.calcuate_attention(passage_reps, question_reps, context_lstm_dim, context_lstm_dim,
                    scope_name="attention", att_type=options.att_type, att_dim=options.att_dim,
                    remove_diagnoal=False, mask1=passage_mask, mask2=question_mask, is_training=is_training, dropout_rate=dropout_rate)
            att_question_contexts = tf.matmul(atten_scores, question_reps)
            (attentive_rep, match_dim) = multi_perspective_match(context_lstm_dim,
                    passage_reps, att_question_contexts, is_training=is_training, dropout_rate=options.dropout_rate,
                    options=options, scope_name='mp-match-att_question')
            all_question_aware_representatins.append(attentive_rep)
            dim += match_dim

        if with_max_attentive_match:
            max_att = cal_max_question_representation(question_reps, relevancy_matrix)
            (max_attentive_rep, match_dim) = multi_perspective_match(context_lstm_dim,
                    passage_reps, max_att, is_training=is_training, dropout_rate=options.dropout_rate,
                    options=options, scope_name='mp-match-max-att')
            all_question_aware_representatins.append(max_attentive_rep)
            dim += match_dim

        all_question_aware_representatins = tf.concat(axis=2, values=all_question_aware_representatins)
    return (all_question_aware_representatins, dim)



    def multi_perspective_match(feature_dim, repres1, repres2, is_training=True, dropout_rate=0.2,
                                options=None, scope_name='mp-match', reuse=False):
        '''
            :param repres1: [batch_size, len, feature_dim]
            :param repres2: [batch_size, len, feature_dim]
            :return:
        '''
        input_shape = tf.shape(repres1)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        matching_result = []
        with tf.variable_scope(scope_name, reuse=reuse):
            match_dim = 0
            if options.with_cosine:
                cosine_value = layer_utils.cosine_distance(repres1, repres2, cosine_norm=False)
                cosine_value = tf.reshape(cosine_value, [batch_size, seq_length, 1])
                matching_result.append(cosine_value)
                match_dim += 1

            if options.with_mp_cosine:
                mp_cosine_params = tf.get_variable("mp_cosine", shape=[options.cosine_MP_dim, feature_dim], dtype=tf.float32)
                mp_cosine_params = tf.expand_dims(mp_cosine_params, axis=0)
                mp_cosine_params = tf.expand_dims(mp_cosine_params, axis=0)
                repres1_flat = tf.expand_dims(repres1, axis=2)
                repres2_flat = tf.expand_dims(repres2, axis=2)
                mp_cosine_matching = layer_utils.cosine_distance(tf.multiply(repres1_flat, mp_cosine_params),
                                                                 repres2_flat,cosine_norm=False)
                matching_result.append(mp_cosine_matching)
                match_dim += options.cosine_MP_dim

        matching_result = tf.concat(axis=2, values=matching_result)
        return (matching_result, match_dim)


def train(sess, saver, train_graph, valid_graph, trainDataStream, devDataStream, options, best_path):
    best_accuracy = -1
    for epoch in range(options.max_epochs):
        print('Train in epoch %d' % epoch)
        # training
        trainDataStream.shuffle()
        num_batch = trainDataStream.get_num_batch()
        start_time = time.time()
        total_loss = 0
        for batch_index in range(num_batch):  # for each batch
            cur_batch = trainDataStream.get_batch(batch_index)
            feed_dict = train_graph.create_feed_dict(cur_batch, is_training=True)
            _, loss_value = sess.run([train_graph.train_op, train_graph.loss], feed_dict=feed_dict)
            total_loss += loss_value
            if batch_index % 100 == 0:
                print('{} '.format(batch_index), end="")
                sys.stdout.flush()

        print()
        duration = time.time() - start_time
        print('Epoch %d: loss = %.4f (%.3f sec)' % (epoch, total_loss / num_batch, duration))
        # evaluation
        start_time = time.time()
        acc = evaluation(sess, valid_graph, devDataStream)
        duration = time.time() - start_time
        print("Accuracy: %.2f" % acc)
        print('Evaluation time: %.3f sec' % (duration))
        if acc>= best_accuracy:
            best_accuracy = acc
            saver.save(sess, best_path)
