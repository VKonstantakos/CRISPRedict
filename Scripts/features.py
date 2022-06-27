import itertools
import pandas as pd


def seq_features(sequence, columns=False, single=False, promoter=None):
    '''
    Calculates selected features from input sequence.

    :parameter sequence: 30-nt sequence(s) to calculate features.
    :parameter columns: option to get the name of the features.
    :parameter single: option to get the features for a single sequence.
    :parameter promoter: gRNA transcription method to compute relevant features ('u6', 't7').
    :return: sequence features.
    '''

    # Transform sequence
    sequence = sequence.upper()

    # Check sequence length
    assert len(sequence) == 30, 'Length of sequence should be 30 nucleotides'

    # Select promoter
    if promoter == 'u6':

        # Overall nucleotide usage

        # GC content
        GC_content = 100*(sequence.count("G") +
                          sequence.count("C"))/len(sequence)

        if GC_content < 35 or GC_content > 80:
            GC_binary = 1
        else:
            GC_binary = 0

        # Dinucleotides
        num_AG = sequence.count("AG")
        num_AA = sequence.count("AA")

        # Trinucleotides
        num_TTT = sequence.count("TTT")
        num_AAA = sequence.count("AAA")
        num_ATT = sequence.count("ATT")
        num_CTT = sequence.count("CTT")

        # Positional features
        # 5' end

        # Position 1
        if sequence[4] == "C":
            c_1 = 1
        else:
            c_1 = 0

        # Position 3
        if sequence[6] == "C":
            c_3 = 1
        else:
            c_3 = 0

        # Middle positions

        a_middle = sequence[11:19].count("A")

        if sequence[13] == "G" or sequence[17] == "G":
            g_middle = 1
        else:
            g_middle = 0

        # Position 16
        if sequence[19] == "A":
            a_16 = 1
        else:
            a_16 = 0

        # Position 13
        if sequence[16] == "A":
            a_13 = 1
        else:
            a_13 = 0

        # Seed region

        # Position 24 (PAM NGG[N])
        if sequence[27] == "G":
            g_24 = 1
        else:
            g_24 = 0

        # Position 21 (PAM [N]GG)
        if sequence[24] == "G":
            g_21 = 1
        else:
            g_21 = 0

        # Position 20
        if sequence[23] == "G":
            g_20 = 1
        else:
            g_20 = 0

        if sequence[23] == "C":
            c_20 = 1
        else:
            c_20 = 0

        # Position 19
        if sequence[22] == "C":
            c_19 = 1
        else:
            c_19 = 0

        # Position 18
        if sequence[21] == "C":
            c_18 = 1
        else:
            c_18 = 0

        # Positions 16-21
        if "T" in sequence[19:25]:
            t_end = 1
        else:
            t_end = 0

        # Dinucleotides position
        if "TG" in sequence[20:24]:
            tg_end = 1
        else:
            tg_end = 0

        if "AC" in sequence[20:23]:
            ac_end = 1
        else:
            ac_end = 0

        if sequence[22:24] == "CA":
            ca_19 = 1
        else:
            ca_19 = 0

        if sequence[27:29] == "CC" or sequence[28:30] == "CC":
            cc_end = 1
        else:
            cc_end = 0

        if sequence[16:18] == "CT":
            ct_13 = 1
        else:
            ct_13 = 0

        if sequence[11:13] == "GT" or sequence[16:18] == "GT" or sequence[20:22] == "GT" or sequence[22:24] == "GT":
            gt_pos = 1
        else:
            gt_pos = 0

        # Trinucleotides position
        if sequence[21:24] == "CCC":
            ccc_18 = 1
        else:
            ccc_18 = 0

        # Motifs
        polyn = ["GGGG", "TTTT", "CCCCC", "AAAAA"]
        motifs = ["GCC", "TT"]

        if any(x in sequence for x in polyn):
            polynuc = 1
        else:
            polynuc = 0

        if any(y in sequence[20:24] for y in motifs):
            motif = 1
        else:
            motif = 0

        features = {"GC binary": GC_binary,
                    "AG count": num_AG, "AA count": num_AA, "TTT count": num_TTT,
                    "AAA count": num_AAA, "ATT count": num_ATT, "CTT count": num_CTT,
                    "G_24": g_24, "G_21": g_21, "G_20": g_20,
                    "C_20": c_20, "C_19": c_19, "C_18": c_18, "C_3": c_3, "C_1": c_1,
                    "A_16": a_16, "A_13": a_13,
                    "A_middle": a_middle, "G_middle": g_middle,
                    "T_end": t_end, "TG_end": tg_end, "AC_end": ac_end,
                    "CA_19": ca_19, "CT_13": ct_13, "GT_pos": gt_pos,
                    "CCC_18": ccc_18,
                    "Motifs": motif, "Polynucleotides": polynuc}

    elif promoter == 't7':

        # Overall nucleotide usage

        # GC content
        GC_content = 100*(sequence.count("G") +
                          sequence.count("C"))/len(sequence)

        # Mononucleotides
        num_G = sequence.count("G")

        # Dinucleotides
        num_GC = sequence.count("GC")

        # Trinucleotides
        num_GAA = sequence.count("GAA")
        num_ATT = sequence.count("ATT")

        # Mononucleotides position
        if sequence[1] == "T":
            t_min3 = 1
        else:
            t_min3 = 0

        if sequence[19] == "T":
            t_16 = 1
        else:
            t_16 = 0

        if sequence[22] == "A":
            a_19 = 1
        else:
            a_19 = 0

        if sequence[27] == "G":
            g_24 = 1
        else:
            g_24 = 0

        # Dinucleotides position
        if sequence[0:2] == "GT":
            gt_min4 = 1
        else:
            gt_min4 = 0

        if sequence[2:4] == "CT":
            ct_min2 = 1
        else:
            ct_min2 = 0

        if sequence[6:8] == "TC":
            tc_3 = 1
        else:
            tc_3 = 0

        if sequence[7:9] == "TC":
            tc_4 = 1
        else:
            tc_4 = 0

        if sequence[10:12] == "GG":
            gg_7 = 1
        else:
            gg_7 = 0

        if sequence[22:24] == "GC":
            gc_19 = 1
        else:
            gc_19 = 0

        if sequence[23:25] == "CT":
            ct_20 = 1
        else:
            ct_20 = 0

        if sequence[28:30] == "AA":
            aa_25 = 1
        else:
            aa_25 = 0

        if sequence[19:21] == "CA" or sequence[27:29] == "CA":
            ca_end = 1
        else:
            ca_end = 0

        if sequence[14:16] == "CC" or sequence[17:19] == "CC" or sequence[20:22] == "CC":
            cc_middle = 1
        else:
            cc_middle = 0

        if sequence[12:14] == "TG" or sequence[15:17] == "TG":
            tg_middle = 1
        else:
            tg_middle = 0

        if sequence[18:20] == 'TC' or sequence[21:23] == 'TC':
            tc_end = 1
        else:
            tc_end = 0

        # Trinucleotides position
        if sequence[4:7] == "GGG":
            ggg_1 = 1
        else:
            ggg_1 = 0

        if sequence[12:15] == "GGG":
            ggg_9 = 1
        else:
            ggg_9 = 0

        if sequence[6:9] == "TGT":
            tgt_3 = 1
        else:
            tgt_3 = 0

        if sequence[20:23] == "GTG":
            gtg_17 = 1
        else:
            gtg_17 = 0

        features = {
            "GC_content": GC_content, "G count": num_G, "GC count": num_GC,
            "GAA count": num_GAA, "ATT count": num_ATT, "T_-3": t_min3,
            "T_16": t_16, "A_19": a_19, "G_24": g_24, "GT_-4": gt_min4,
            "CT_-2": ct_min2, "TC_3": tc_3, "TC_4": tc_4, "GG_7": gg_7,
            "GC_19": gc_19, "CT_20": ct_20, "AA_25": aa_25, "CA_end": ca_end,
            "CC_middle": cc_middle, "TC_end": tc_end, "TG_middle": tg_middle,
            "GGG_1": ggg_1, "GGG_9": ggg_9, "TGT_3": tgt_3, "GTG_17": gtg_17
        }

    else:
        raise ValueError(
            "No promoter was selected. Select a valid promoter ('u6' or 't7') to calculate the relevant features.")

    if columns == True:
        return list(features.keys())
    elif single == True:
        return pd.DataFrame.from_dict(features, orient='index').transpose()
    else:
        return list(features.values())


def seq_train(X, promoter=None):
    '''
    Calculates selected features and prepares dataset for training.

    :parameter X: feature set to transform.
    :parameter promoter: gRNA transcription method to compute relevant features ('u6', 't7').
    :return: transformed feature set.
    '''

    if promoter:
        # Get the name of the features
        cols = seq_features(X[0], columns=True, promoter=promoter)

        # Calculate sequence features
        cols_values = X.apply(seq_features, args=(
            False, False, promoter)).to_frame()

        return pd.DataFrame(cols_values.iloc[:, 0].to_list(), columns=cols)

    else:
        raise ValueError(
            "No promoter was selected. Select a valid promoter ('u6' or 't7') to calculate the relevant features.")


def get_nucleotides(length, dna_code=['A', 'T', 'C', 'G']):
    '''
    Calculates combinations of nucleotides for a specified length.

    :parameter length: length of combined nucleotides.
    :parameter dna_code: DNA bases to combine.
    :return: all combinations.
    '''

    nucl = ["".join(i) for i in itertools.product(dna_code, repeat=length)]

    return nucl


def get_counts(sequence, max_length=3):
    '''
    Counts the different combinations of nucleotides up to a specified length.

    :parameter sequence: sequence to analyse.
    :parameter max_length: maximum length of combined nucleotides. Default is 3.
    :return: dictionary with each nucleotide and the respective count.
    '''

    # Initialize empty dictionary
    nuc_count = {}

    # For each length up to max length
    for length in range(1, max_length+1):

        # Create all combinations
        for x in get_nucleotides(length):

            # Count each combination and store it
            nuc_count[x] = sequence.count(x)

    return nuc_count


def get_positions(sequence, max_length=3):
    '''
    Calculates the position for each nucleotide combination.

    :parameter sequence: sequence to analyse.
    :parameter max_length: maximum length of combined nucleotides. Default is 3.
    :return: position-specific nucleotides.
    '''

    # Initialize empty list and dictionary
    positions = []
    nuc_pos = {}

    # Create all the nucleotide combinations for each position
    for length in range(1, max_length+1):
        for num in range(1, len(sequence)+2-length):
            positions.extend([i + '_' + str(num)
                              for i in get_nucleotides(length)])

    # Populate dictionary with all features equal to zero
    for x in positions:
        nuc_pos[x] = 0

    # For each length up to max length
    for length in range(1, max_length+1):

        # For each sequence position
        for pos in range(0, len(sequence)+1-length):

            # Count the nucleotide for that position
            nuc_pos[sequence[pos:pos+length] + '_' + str(pos+1)] += 1

    return nuc_pos


def misc_features(sequence):
    '''
    Calculates miscellaneous features (other than count/position) from input sequence.

    :parameter sequence: 30-nt sequence.
    :return: dictionary with calculated features.
    '''

    # Check sequence length
    assert len(sequence) == 30, 'Length of sequence should be 30 nucleotides'

    # GC content
    GC_content = 100*(sequence.count('G') + sequence.count('C'))/len(sequence)

    if GC_content < 35 or GC_content > 80:
        GC_binary = 1.0
    else:
        GC_binary = 0.0

    # Proximal GC content
    prox_GC = 100 * (sequence[4:14].count('G') +
                     sequence[4:14].count('C'))/len(sequence[4:14])

    if prox_GC > 70:
        prox_GC_binary = 1.0
    else:
        prox_GC_binary = 0.0

    # Middle positions
    a_middle = sequence[11:19].count('A')

    # Motifs
    polyn = ["GGGG", "TTTT", "CCCCC", "AAAAA"]
    motifs = ["GCC", "TT"]

    if any(x in sequence for x in polyn) or any(y in sequence[20:24] for y in motifs):
        motif = 1.0
    else:
        motif = 0.0

    features = {"GC content": GC_content, "GC binary": GC_binary,
                "Proximal GC content": prox_GC, "Proximal GC binary": prox_GC_binary,
                "A in the middle": a_middle, "Motifs": motif}

    return features


def get_features(sequence, max_length=3, columns=False, single=False):
    '''
    Calculates all features from input sequence.

    :parameter sequence: 30-nt sequence(s) to calculate features.
    :parameter max_length: maximum length of combined nucleotides. Default is 3.
    :parameter columns: option to get the name of the features.
    :parameter single: option to get the features for a single sequence.
    :return: sequence features.
    '''

    # Check sequence length
    assert len(sequence) == 30, 'Length of sequence should be 30 nucleotides'

    # Calculate count/position/miscellaneous sequence features
    count_dict = get_counts(sequence, max_length)
    pos_dict = get_positions(sequence, max_length)
    misc_dict = misc_features(sequence)
    features_dict = {**count_dict, **pos_dict, **misc_dict}

    if columns == True:
        return list(features_dict.keys())
    elif single == True:
        return pd.DataFrame.from_dict(features_dict, orient='index').transpose()
    else:
        return list(features_dict.values())


def feature_train(X):
    '''
    Calculates all features and prepares dataset for training.

    :parameter X: feature set to transform.
    :return: transformed feature set.
    '''

    # Get the name of the features
    cols = get_features(X[0], columns=True)

    # Calculate sequence features
    cols_values = X.apply(get_features).to_frame()

    return pd.DataFrame(cols_values.iloc[:, 0].to_list(), columns=cols)
